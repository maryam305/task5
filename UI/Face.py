import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import json
from datetime import datetime

# ==========================================
# Application Settings
# ==========================================
WINDOW_NAME = "Snap Filter Pro - With Opacity Slider"
ASSETS_DIR = "assets"

mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

class FaceMorphApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        self.asset_loader_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2
        )

        self.segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        self.categories = {} 
        self.category_names = []
        self.current_category = None 
        self.selected_asset = None   
        
        self.frame_w = 0
        self.frame_h = 0

        # ### متغيرات الشريط الجديد ###
        self.opacity = 1.0 # القيمة الافتراضية (ظاهر بالكامل)
        self.is_dragging_slider = False
        self.slider_rect = (0, 0, 0, 0) # سيتم تحديدها لاحقاً (x, y, w, h)
        
        print("Loading assets...")
        self.load_assets_by_folders()
        
        cv2.namedWindow(WINDOW_NAME)
        # ### نحتاج الآن لتعقب حركة الماوس أيضاً وليس فقط النقر ###
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse_click)

    def load_assets_by_folders(self):
        # (نفس الكود السابق تماماً، لم يتغير شيء هنا)
        if not os.path.exists(ASSETS_DIR):
            os.makedirs(ASSETS_DIR)
            return
        folders = [f for f in os.listdir(ASSETS_DIR) if os.path.isdir(os.path.join(ASSETS_DIR, f))]
        for folder in folders:
            folder_path = os.path.join(ASSETS_DIR, folder)
            self.categories[folder] = []
            extensions = ('*.png', '*.webp', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
            all_files = []
            for ext in extensions: all_files.extend(glob.glob(os.path.join(folder_path, ext)))
            unique_files = list(set(all_files))
            for fpath in unique_files:
                try:
                    asset_data = self.process_single_asset(fpath)
                    if asset_data: self.categories[folder].append(asset_data)
                except: pass
            if self.categories[folder]: self.category_names.append(folder)

    def process_single_asset(self, fpath):
        # (نفس الكود السابق تماماً)
        img_original = cv2.imread(fpath)
        if img_original is None: return None
        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        res = self.asset_loader_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            h, w = img_rgb.shape[:2]
            temp_img = cv2.resize(img_rgb, (w*2, h*2))
            res = self.asset_loader_mesh.process(temp_img)
            if not res.multi_face_landmarks: return None
        seg_res = self.segmenter.process(img_rgb)
        mask_val = seg_res.segmentation_mask if seg_res.segmentation_mask is not None else np.ones(img_original.shape[:2], dtype=np.float32)
        mask = (mask_val > 0.4)
        alpha_mask = np.zeros(img_original.shape[:2], dtype=np.uint8)
        alpha_mask[mask] = 255
        alpha_mask = cv2.GaussianBlur(alpha_mask, (3, 3), 0)
        b, g, r = cv2.split(img_original)
        img_rgba = cv2.merge((b, g, r, alpha_mask))
        h, w = img_original.shape[:2]
        landmarks = np.array([[int(p.x * w), int(p.y * h)] for p in res.multi_face_landmarks[0].landmark], dtype=np.int32)
        thumb = cv2.resize(img_rgba, (60, 60))
        mask_c = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(mask_c, (30, 30), 30, 255, -1)
        tb, tg, tr, ta = cv2.split(thumb)
        ta = cv2.bitwise_and(ta, ta, mask=mask_c)
        thumb_final = cv2.merge((tb, tg, tr, ta))
        boundary = np.array([[0,0], [w//2,0], [w-1,0], [w-1,h//2], [w-1,h-1], [w//2,h-1], [0,h-1], [0,h//2]])
        full_lm = np.vstack((landmarks, boundary))
        tri = self.calculate_delaunay(full_lm)
        return {"img": img_rgba, "lm": full_lm, "tri": tri, "thumb": thumb_final}

    def calculate_delaunay(self, points):
        # (نفس الكود السابق)
        rect = (0, 0, 4000, 4000)
        subdiv = cv2.Subdiv2D(rect)
        for p in points: subdiv.insert((float(p[0]), float(p[1])))
        pt_dict = {(p[0], p[1]): i for i, p in enumerate(points)}
        triangles = []
        for t in subdiv.getTriangleList():
            pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
            if all(pt in pt_dict for pt in pts): triangles.append([pt_dict[pt] for pt in pts])
        return triangles

    # ### تعديل جوهري في التعامل مع الماوس لدعم السحب ###
    def on_mouse_click(self, event, x, y, flags, param):
        # 1. التعامل مع الضغط (بداية السحب أو النقر)
        if event == cv2.EVENT_LBUTTONDOWN:
            # التحقق هل الضغط تم داخل منطقة الشريط؟
            sx, sy, sw, sh = self.slider_rect
            # نضيف هامش بسيط (15 بكسل) حول الشريط لتسهيل اللمس
            if sx - 15 <= x <= sx + sw + 15 and sy - 15 <= y <= sy + sh + 15:
                self.is_dragging_slider = True
                # حساب القيمة مباشرة عند الضغط
                self.opacity = np.clip((x - sx) / sw, 0.0, 1.0)
                return # نخرج حتى لا يتعارض مع الأزرار الأخرى

            # (باقي أكواد الأزرار العادية - تعمل فقط إذا لم نكن نضغط على الشريط)
            if np.linalg.norm(np.array([x, y]) - np.array([self.frame_w//2, self.frame_h-50])) < 35:
                self.save_snapshot()
                return

            sx_ui, sy_ui = 20, self.frame_h - 130
            if sx_ui < x < sx_ui+60 and sy_ui < y < sy_ui+60:
                if self.current_category: self.current_category = None
                else: self.selected_asset = None
                return

            curr_x = sx_ui + 80
            if self.current_category is None:
                for cat in self.category_names:
                    if curr_x < x < curr_x+60 and sy_ui < y < sy_ui+60:
                        self.current_category = cat
                        return
                    curr_x += 75
            else:
                for asset in self.categories[self.current_category]:
                    if curr_x < x < curr_x+60 and sy_ui < y < sy_ui+60:
                        self.selected_asset = asset
                        return
                    curr_x += 75

        # 2. التعامل مع تحريك الماوس (أثناء السحب)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging_slider:
                sx, sy, sw, sh = self.slider_rect
                # حساب الموقع النسبي للماوس داخل الشريط (بين 0.0 و 1.0)
                self.opacity = np.clip((x - sx) / sw, 0.0, 1.0)

        # 3. التعامل مع رفع الزر (انتهاء السحب)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging_slider = False

    def draw_ui(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.frame_h - 145), (self.frame_w, self.frame_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # --- رسم الأزرار السفلية (نفس الكود السابق) ---
        sx, sy = 20, self.frame_h - 130
        is_in_cat = (self.current_category is not None)
        cv2.circle(frame, (sx+30, sy+30), 30, (255, 255, 255), 2)
        txt = "BACK" if is_in_cat else "OFF"
        cv2.putText(frame, txt, (sx+15, sy+38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        curr_x = sx + 80
        if self.current_category is None:
            for cat in self.category_names:
                cover = self.categories[cat][0]["thumb"]
                self.draw_icon(frame, cover, curr_x, sy, False)
                cv2.putText(frame, cat[:10], (curr_x, sy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                curr_x += 75
        else:
            for asset in self.categories[self.current_category]:
                is_sel = (self.selected_asset is asset)
                self.draw_icon(frame, asset["thumb"], curr_x, sy, is_sel)
                curr_x += 75

        cv2.circle(frame, (self.frame_w//2, self.frame_h-50), 32, (255, 255, 255), 3)

        # ### رسم شريط الشفافية (Slider) ###
        if self.selected_asset is not None: # نرسمه فقط إذا كان هناك فلتر مختار
            # إعدادات مكان وحجم الشريط (فوق الشريط السفلي على اليمين)
            slider_w = 200
            slider_h = 8
            slider_x = self.frame_w - slider_w - 30
            slider_y = self.frame_h - 180
            
            # تخزين المساحة لاستخدامها في الماوس
            self.slider_rect = (slider_x, slider_y, slider_w, slider_h)

            # 1. رسم خلفية الشريط (الخط الرمادي)
            cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (100, 100, 100), -1, cv2.LINE_AA)
            
            # 2. رسم الجزء الممتلئ (بناءً على القيمة)
            filled_w = int(slider_w * self.opacity)
            cv2.rectangle(frame, (slider_x, slider_y), (slider_x + filled_w, slider_y + slider_h), (0, 255, 255), -1, cv2.LINE_AA)

            # 3. رسم المقبض (الدائرة)
            knob_x = slider_x + filled_w
            knob_y = slider_y + slider_h // 2
            cv2.circle(frame, (knob_x, knob_y), 12, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (knob_x, knob_y), 14, (0, 150, 150), 2, cv2.LINE_AA) # إطار للمقبض

            # إضافة نص توضيحي
            cv2.putText(frame, f"Opacity: {int(self.opacity*100)}%", (slider_x, slider_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def draw_icon(self, frame, thumb, x, y, is_selected):
        if x + 60 > self.frame_w: return
        roi = frame[y:y+60, x:x+60]
        img, alpha = thumb[:,:,:3], thumb[:,:,3]/255.0
        a3 = np.dstack([alpha]*3)
        frame[y:y+60, x:x+60] = (img * a3 + roi * (1-a3)).astype(np.uint8)
        color = (0, 255, 0) if is_selected else (200, 200, 200)
        cv2.circle(frame, (x+30, y+30), 31, color, 2 if is_selected else 1)

    def get_user_boundary_points(self, user_lm, frame_w, frame_h):
        # (نفس الكود السابق)
        x_min, y_min = np.min(user_lm, axis=0)
        x_max, y_max = np.max(user_lm, axis=0)
        wf, hf = x_max - x_min, y_max - y_min
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        s = 0.6
        nx1, nx2 = max(0, cx-int(wf*(0.5+s))), min(frame_w, cx+int(wf*(0.5+s)))
        ny1, ny2 = max(0, cy-int(hf*(0.5+s))), min(frame_h, cy+int(hf*(0.5+s)))
        return np.array([[nx1,ny1],[cx,ny1],[nx2,ny1],[nx2,cy],[nx2,ny2],[cx,ny2],[nx1,ny2],[nx1,cy]], dtype=np.int32)

    # ### تعديل جوهري لتطبيق قيمة الشفافية ###
    def warp_face_transparent(self, frame, asset, user_lm):
        src_img, src_pts, tris = asset["img"], asset["lm"], asset["tri"]
        user_pts = np.vstack((user_lm, self.get_user_boundary_points(user_lm, self.frame_w, self.frame_h)))
        
        warped_rgba = np.zeros((self.frame_h, self.frame_w, 4), dtype=np.uint8)

        for tri in tris:
            ps = [src_pts[i] for i in tri]
            pt = [user_pts[i] for i in tri]
            r1, r2 = cv2.boundingRect(np.float32(ps)), cv2.boundingRect(np.float32(pt))
            if r1[2]<=0 or r1[3]<=0 or r2[2]<=0 or r2[3]<=0: continue
            ts = [(p[0]-r1[0], p[1]-r1[1]) for p in ps]
            tt = [(p[0]-r2[0], p[1]-r2[1]) for p in pt]
            mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(tt), 255)
            img1 = src_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
            mat = cv2.getAffineTransform(np.float32(ts), np.float32(tt))
            img2 = cv2.warpAffine(img1, mat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            y1, y2, x1, x2 = r2[1], r2[1]+r2[3], r2[0], r2[0]+r2[2]
            if y1<0 or x1<0 or y2>self.frame_h or x2>self.frame_w: continue
            target = warped_rgba[y1:y2, x1:x2]
            target[mask>0] = img2[mask>0]

        # --- هنا يتم تطبيق الشفافية ---
        rgb = warped_rgba[:,:,:3]
        # الحصول على قناة الألفا الأصلية وتنعيمها
        base_alpha = cv2.GaussianBlur(warped_rgba[:,:,3]/255.0, (3,3), 0)
        
        # ضرب الألفا في قيمة الشريط (self.opacity) للتحكم في الظهور
        final_alpha = base_alpha * self.opacity 
        
        a3 = np.dstack([final_alpha]*3)
        
        # الدمج النهائي باستخدام الألفا المعدلة
        foreground = rgb.astype(np.float32)
        background = frame.astype(np.float32)
        final_img = foreground * a3 + background * (1.0 - a3)
        
        return final_img.astype(np.uint8)

    def save_snapshot(self):
        fn = f"snap_{datetime.now().strftime('%H%M%S')}.png"
        cv2.imwrite(fn, self.clean_frame)
        print(f"Snapshot saved: {fn}")

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            self.frame_h, self.frame_w = frame.shape[:2]
            
            output = frame.copy()
            res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if res.multi_face_landmarks:
                pts = np.array([[int(p.x * self.frame_w), int(p.y * self.frame_h)] for p in res.multi_face_landmarks[0].landmark], dtype=np.int32)
                if self.selected_asset:
                    # يتم الآن تطبيق الشفافية داخل هذه الدالة
                    output = self.warp_face_transparent(frame, self.selected_asset, pts)
            
            self.clean_frame = output.copy()
            self.draw_ui(output)
            cv2.imshow(WINDOW_NAME, output)
            
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: break
            
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceMorphApp()
    app.run()