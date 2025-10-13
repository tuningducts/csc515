import argparse, cv2
import numpy as np 
import mediapipe as mp
import numpy as np

# face landmarks from media pipe documentation
LEFT_EYE = [263, 362, 387, 386, 385, 384, 398, 373, 374, 380]
RIGHT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153]
MOUTH_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308]
MOUTH_INNER = [78,95,88,178,87,14,317,402,318,324,308,415]  # inner lips ring (approx)

def facemesh_pts(img_bgr):
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                         max_num_faces=1, min_detection_confidence=0.5) as fm:
        res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks: return None
        h, w = img_bgr.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        return np.int32([(p.x*w, p.y*h) for p in lm])  # (468,2)

def eye_mouth_triplet(pts):
    le = pts[LEFT_EYE].mean(axis=0); re = pts[RIGHT_EYE].mean(axis=0); mouth = pts[MOUTH_OUTER].mean(axis=0)
    return np.float32([le, re, mouth])

def delaunay_tri_indices(img_shape, points):
    H, W = img_shape[:2]
    subdiv = cv2.Subdiv2D((0,0,W,H))
    for (x,y) in points:
        if 0 <= x < W and 0 <= y < H: subdiv.insert((float(x), float(y)))
    tris = subdiv.getTriangleList()
    ptsf = points.astype(np.float32)
    idx_tris = []
    for x1,y1,x2,y2,x3,y3 in tris:
        tri = [np.array([x1,y1],np.float32), np.array([x2,y2],np.float32), np.array([x3,y3],np.float32)]
        if any(not(0<=p[0]<W and 0<=p[1]<H) for p in tri): continue
        i0 = int(np.argmin(np.sum((ptsf - tri[0])**2, axis=1)))
        i1 = int(np.argmin(np.sum((ptsf - tri[1])**2, axis=1)))
        i2 = int(np.argmin(np.sum((ptsf - tri[2])**2, axis=1)))
        if len({i0,i1,i2})==3: idx_tris.append((i0,i1,i2))
    return idx_tris


def warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(t_src); r2 = cv2.boundingRect(t_dst)
    t1 = np.float32([[t_src[0][0]-r1[0], t_src[0][1]-r1[1]],
                     [t_src[1][0]-r1[0], t_src[1][1]-r1[1]],
                     [t_src[2][0]-r1[0], t_src[2][1]-r1[1]]])
    t2 = np.float32([[t_dst[0][0]-r2[0], t_dst[0][1]-r2[1]],
                     [t_dst[1][0]-r2[0], t_dst[1][1]-r2[1]],
                     [t_dst[2][0]-r2[0], t_dst[2][1]-r2[1]]])
    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(t2), 255)
    src_patch = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    M = cv2.getAffineTransform(t1, t2)
    warped = cv2.warpAffine(src_patch, M, (r2[2], r2[3]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    roi = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    np.copyto(roi, warped, where=mask[...,None].astype(bool))

def face_hull_mask(shape_hw, pts, shrink_px=8, blur_ks=21, blur_sigma=8):
    H, W = shape_hw
    mask = np.zeros((H, W), np.uint8)
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if shrink_px>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*shrink_px+1, 2*shrink_px+1))
        mask = cv2.erode(mask, k, 1)
    mask = cv2.GaussianBlur(mask, (blur_ks, blur_ks), blur_sigma)
    return mask

def cutout_mouth(mask_u8, pts, inflate_px=2):
    mouth = pts[MOUTH_INNER].astype(np.int32)
    hole = np.zeros_like(mask_u8)
    cv2.fillConvexPoly(hole, cv2.convexHull(mouth), 255)
    if inflate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inflate_px+1, 2*inflate_px+1))
        hole = cv2.dilate(hole, k, 1)
    mouth_hole = cv2.GaussianBlur(hole, (11,11), 5)
    # subtract with safety (float math then clip)
    out = (mask_u8.astype(np.float32) - mouth_hole.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)

def swap_face(src, dst, blend="alpha", remove_teeth=True):
    src_pts = facemesh_pts(src); dst_pts = facemesh_pts(dst)
    if src_pts is None or dst_pts is None:
        print("[WARN] face not found"); return None

    H, W = dst.shape[:2]

   
    M = cv2.getAffineTransform(eye_mouth_triplet(src_pts), eye_mouth_triplet(dst_pts))
    src_pre = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    src_pts_w = cv2.transform(src_pts[None,...].astype(np.float32), M)[0].astype(np.int32)

  
    tris = delaunay_tri_indices(dst.shape, dst_pts)
    canvas = np.zeros_like(dst)
    for i0,i1,i2 in tris:
        t_dst = np.float32([dst_pts[i0], dst_pts[i1], dst_pts[i2]])
        t_src = np.float32([src_pts_w[i0], src_pts_w[i1], src_pts_w[i2]])
        warp_triangle(src_pre, canvas, t_src, t_dst)

   
    alpha_u8 = face_hull_mask((H, W), dst_pts, shrink_px=10, blur_ks=23, blur_sigma=9)
    if remove_teeth:
        alpha_u8 = cutout_mouth(alpha_u8, dst_pts, inflate_px=3)

 
    content = (cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    alpha_u8 = cv2.bitwise_and(alpha_u8, content)

    ys, xs = np.where(alpha_u8 > 0)
    if len(xs)==0: print("[WARN] empty mask"); return None
    center = (int(np.mean(xs)), int(np.mean(ys)))

    if blend == "seamless":
        out = cv2.seamlessClone(canvas, dst, alpha_u8, center, flags=cv2.MIXED_CLONE)
    else:
        a = (alpha_u8.astype(np.float32)/255.0)[..., None]
        out = (dst.astype(np.float32)*(1-a) + canvas.astype(np.float32)*a).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser(description="Face swap (MediaPipe + Delaunay, mouth-safe)")
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--out", default="swapped.jpg")
    ap.add_argument("--blend", choices=["alpha","seamless"], default="alpha")
    ap.add_argument("--keep_teeth", action="store_true", help="If set, do NOT cut out mouth (paste teeth).")
    args = ap.parse_args()

    src = cv2.imread(args.src); dst = cv2.imread(args.dst)
    if src is None or dst is None: print("[ERROR] bad path(s)"); return

    out = swap_face(src, dst, blend=args.blend, remove_teeth=not args.keep_teeth)
    if out is None: print("[ERROR] swap failed"); return
    cv2.imwrite(args.out, out); print(f"[OK] Wrote {args.out}")

if __name__ == "__main__":
    main()
