# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import tensorflow as tf

# # # ==========================
# # # 1️⃣ SETUP & LENS LOAD
# # # ==========================
# # TFLITE_PATH = "iris_pure_float32.tflite"
# # interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
# # interpreter.allocate_tensors()
# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()

# # LENS_PATH = "images/lenses.png" # Aapki lens texture image
# # lens_img = cv2.imread(LENS_PATH, cv2.IMREAD_UNCHANGED)

# # # ==========================
# # # 2️⃣ EXACT OVERLAY LOGIC (No Black Circle)
# # # ==========================
# # # def apply_exact_overlay(crop, mask, lens_texture):
# # #     h, w = crop.shape[:2]
    
# # #     # 1. Lens ko crop size par laein
# # #     lens_res = cv2.resize(lens_texture, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
# # #     # 2. Extract Color and Alpha
# # #     if lens_res.shape[2] == 4:
# # #         lens_bgr = lens_res[:, :, :3]
# # #         lens_alpha = lens_res[:, :, 3].astype(float) / 255.0
# # #     else:
# # #         lens_bgr = lens_res
# # #         lens_alpha = np.ones((h, w), dtype=float)

# # #     # 3. Mask ko 'Dilate' karein (taaki lens asli iris ko poora dhak le)
# # #     kernel = np.ones((3,3), np.uint8)
# # #     mask = cv2.dilate(mask, kernel, iterations=1)
# # #     mask_norm = mask.astype(float) / 255.0

# # #     # 4. Final Alpha: Isme hum asli iris ko 0 priority denge jahan mask hai
# # #     alpha_3d = np.expand_dims(mask_norm * lens_alpha, axis=-1)

# # #     # 5. OVERLAY (Lens will be on TOP)
# # #     # Jahan alpha 1 hai, wahan sirf lens dikhega, asli aankh bilkul nahi
# # #     out = (crop.astype(float) * (1.0 - alpha_3d)) + (lens_bgr.astype(float) * alpha_3d)
    
# # #     return np.clip(out, 0, 255).astype(np.uint8)

# # def apply_png_lens_overlay(crop, mask, lens_texture):
# #     h, w = crop.shape[:2]
    
# #     # 1. Lens resize (LANCZOS behtareen quality ke liye)
# #     lens_res = cv2.resize(lens_texture, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
# #     if lens_res.shape[2] == 4:
# #         lens_bgr = lens_res[:, :, :3]
# #         lens_alpha = lens_res[:, :, 3].astype(float) / 255.0
# #     else:
# #         lens_bgr = lens_res
# #         lens_alpha = np.ones((h, w), dtype=float)

# #     # 2. 🔥 HOLE FILLING LOGIC (Isse black iris khatam hoga)
# #     # Mask ke andar ke har kism ke holes/gaps ko bharne ke liye
# #     kernel = np.ones((7,7), np.uint8)
# #     mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Gap bharta hai
# #     mask_solid = cv2.dilate(mask_filled, kernel, iterations=1)    # Edges cover karta hai
    
# #     mask_norm = mask_solid.astype(float) / 255.0
# #     # Edges ko halka sa smooth karein
# #     mask_norm = cv2.GaussianBlur(mask_norm, (5, 5), 0)

# #     # 3. Alpha priority set karein
# #     final_alpha = mask_norm * lens_alpha
# #     alpha_3d = np.expand_dims(final_alpha, axis=-1)

# #     # 4. Force Overlay (Asli iris ko bilkul chupa dein)
# #     # Jahan alpha hai wahan sirf lens ka texture aayega
# #     foreground = lens_bgr.astype(float) * alpha_3d
# #     background = crop.astype(float) * (1.0 - alpha_3d)
    
# #     result = cv2.add(foreground, background)
# #     return np.clip(result, 0, 255).astype(np.uint8)
# # # ==========================
# # # 3️⃣ MODEL PREDICTION
# # # ==========================
# # def predict_mask(crop):
# #     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
# #     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
# #     interpreter.invoke()
# #     pred = interpreter.get_tensor(output_details[0]['index'])[0]
# #     mask = (np.squeeze(pred) > 0.5).astype(np.uint8) * 255
# #     return cv2.resize(mask, (crop.shape[1], crop.shape[0]))


# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(
# #     refine_landmarks=True, 
# #     min_detection_confidence=0.7, 
# #     min_tracking_confidence=0.7
# # )

# # cap = cv2.VideoCapture(0)

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret: break
# #     frame = cv2.flip(frame, 1)
# #     h, w = frame.shape[:2]
    
# #     # Mediapipe processing
# #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     results = face_mesh.process(rgb_frame)
    
# #     if results.multi_face_landmarks:
# #         landmarks = results.multi_face_landmarks[0].landmark
        
# #         # # Distance calculate karein
# #         # # 1. Face distance calculate karein (Landmarks 468 aur 473 se)
# #         # p1 = np.array([landmarks[468].x, landmarks[468].y])
# #         # p2 = np.array([landmarks[473].x, landmarks[473].y])
# #         # dist = np.linalg.norm(p1 - p2)
# #         # # 2. DYNAMIC BOX: Door jane par box size chota karein 
# #         # # Isse UNet model ko hamesha 'Big Iris' nazar aayegi
# #         # b = int(dist * w * 0.75) 
# #         # b = max(40, min(b, 100)) # Door: 40px, Kareeb: 100px
# #         # Distance based dynamic box
# #         p1 = np.array([landmarks[468].x, landmarks[468].y])
# #         p2 = np.array([landmarks[473].x, landmarks[473].y])
# #         dist = np.linalg.norm(p1 - p2)
        
# #         # Door hone par (dist chota), box ko thoda 'tight' rakhein
# #         # Isse model ko iris bada dikhega aur mask solid banega
# #         b = int(dist * w * 0.85) 
# #         b = max(35, min(b, 110)) # Minimum 35px ka box door se bhi iris ko cover karega


# #         for idx in [468, 473]:
# #             cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            
# #             # Slicing with boundary safety
# #             y1, y2, x1, x2 = max(0, cy-b), min(h, cy+b), max(0, cx-b), min(w, cx+b)
# #             crop = frame[y1:y2, x1:x2].copy()
            
# #             if crop.size > 0:
# #                 # Mask prediction (Aapka predict_mask function yahan call hoga)
# #                 mask = predict_mask(crop) 
                
# #                 # Apply the Overlay
# #                 result_eye = apply_png_lens_overlay(crop, mask, lens_img)
                
# #                 # Check shape compatibility before putting back
# #                 if result_eye.shape == frame[y1:y2, x1:x2].shape:
# #                     frame[y1:y2, x1:x2] = result_eye

# #     cv2.imshow("TTDEye Realistic Overlay", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'): break

# # cap.release()
# # cv2.destroyAllWindows()


# # import cv2
# # import mediapipe as mp
# # import numpy as np

# # # =======================
# # # Load transparent lens
# # # =======================
# # lens_png = cv2.imread("images/lenses.png", cv2.IMREAD_UNCHANGED)
# # assert lens_png is not None, "lens.png not found!"

# # # =======================
# # # MediaPipe setup
# # # =======================
# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(
# #     max_num_faces=1,
# #     refine_landmarks=True,
# #     min_detection_confidence=0.5,
# #     min_tracking_confidence=0.5
# # )

# # # Iris landmark indices
# # LEFT_IRIS  = [474, 475, 476, 477]
# # RIGHT_IRIS = [469, 470, 471, 472]

# # # =======================
# # # Helper: overlay PNG
# # # =======================
# # def overlay_png(bg, png, center, radius):
# #     png = cv2.resize(png, (radius * 2, radius * 2))

# #     x, y = center
# #     h, w = png.shape[:2]

# #     x1 = max(0, x - radius)
# #     y1 = max(0, y - radius)
# #     x2 = min(bg.shape[1], x + radius)
# #     y2 = min(bg.shape[0], y + radius)

# #     png = png[0:y2 - y1, 0:x2 - x1]

# #     if png.shape[2] == 4:
# #         alpha = png[:, :, 3] / 255.0
# #         for c in range(3):
# #             bg[y1:y2, x1:x2, c] = (
# #                 alpha * png[:, :, c] +
# #                 (1 - alpha) * bg[y1:y2, x1:x2, c]
# #             )
# #     return bg

# # # =======================
# # # Camera
# # # =======================
# # cap = cv2.VideoCapture(0)

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     frame = cv2.flip(frame, 1)
# #     h, w, _ = frame.shape

# #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     results = face_mesh.process(rgb)

# #     if results.multi_face_landmarks:
# #         lm = results.multi_face_landmarks[0].landmark

# #         for iris in [LEFT_IRIS, RIGHT_IRIS]:
# #             pts = np.array(
# #                 [(int(lm[i].x * w), int(lm[i].y * h)) for i in iris]
# #             )

# #             center = pts.mean(axis=0).astype(int)
# #             radius = int(np.linalg.norm(pts[0] - pts[2]) / 2.2)

# #             if radius > 5:
# #                 frame = overlay_png(frame, lens_png, tuple(center), radius)

# #     cv2.imshow("Virtual Contact Lens (MediaPipe)", frame)
# #     if cv2.waitKey(1) & 0xFF == 27:
# #         break

# # cap.release()
# # cv2.destroyAllWindows()


# #abdullah code
# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import tensorflow as tf

# # # ==========================
# # # 1️⃣ SETUP & LENS LOAD
# # # ==========================
# # TFLITE_PATH = "iris_pure_float32.tflite"
# # interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
# # interpreter.allocate_tensors()
# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()

# # LENS_PATH = "images/test1.png"  # Aapki lens texture image
# # lens_img = cv2.imread(LENS_PATH, cv2.IMREAD_UNCHANGED)

# # # ==========================
# # # 2️⃣ IMPROVED OVERLAY LOGIC
# # # ==========================
# # def apply_png_lens_overlay(crop, mask, lens_texture, distance_factor):
# #     """
# #     Improved overlay function with distance-aware processing
# #     distance_factor: 0.0 (far) to 1.0 (close)
# #     """
# #     h, w = crop.shape[:2]
    
# #     # 1. High-quality lens resize
# #     lens_res = cv2.resize(lens_texture, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
# #     if lens_res.shape[2] == 4:
# #         lens_bgr = lens_res[:, :, :3]
# #         lens_alpha = lens_res[:, :, 3].astype(float) / 255.0
# #     else:
# #         lens_bgr = lens_res
# #         lens_alpha = np.ones((h, w), dtype=float)

# #     # 2. 🔥 ADAPTIVE MORPHOLOGY - Distance ke basis par adjust karo
# #     # Door hone par zyada aggressive dilation
# #     kernel_size = 9 if distance_factor < 0.5 else 7
# #     kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
# #     # Multiple passes for far distances
# #     iterations = 2 if distance_factor < 0.5 else 1
    
# #     # Hole filling
# #     mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
# #     # Boundary expansion
# #     mask_solid = cv2.dilate(mask_filled, kernel, iterations=iterations)
    
# #     # 3. Circular enhancement - Iris naturally circular hai
# #     # Find contours and fill them
# #     contours, _ = cv2.findContours(mask_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if contours:
# #         # Largest contour lein (iris hoga)
# #         largest_contour = max(contours, key=cv2.contourArea)
# #         # Filled circular mask banain
# #         mask_circular = np.zeros_like(mask_solid)
# #         cv2.drawContours(mask_circular, [largest_contour], -1, 255, -1)
# #         # Combine original mask with circular mask
# #         mask_solid = cv2.bitwise_or(mask_solid, mask_circular)
    
# #     mask_norm = mask_solid.astype(float) / 255.0
    
# #     # 4. Adaptive smoothing - Door hone par zyada smooth
# #     blur_size = 7 if distance_factor < 0.5 else 5
# #     if blur_size % 2 == 0:
# #         blur_size += 1
# #     mask_norm = cv2.GaussianBlur(mask_norm, (blur_size, blur_size), 0)

# #     # 5. Alpha composition
# #     final_alpha = mask_norm * lens_alpha
# #     alpha_3d = np.expand_dims(final_alpha, axis=-1)

# #     # 6. Blending with edge feathering
# #     foreground = lens_bgr.astype(float) * alpha_3d
# #     background = crop.astype(float) * (1.0 - alpha_3d)
    
# #     result = cv2.add(foreground, background)
# #     return np.clip(result, 0, 255).astype(np.uint8)

# # # ==========================
# # # 3️⃣ IMPROVED MODEL PREDICTION
# # # ==========================
# # def predict_mask(crop, distance_factor):
# #     """
# #     Enhanced mask prediction with preprocessing
# #     """
# #     # Pre-processing: Contrast enhancement for better segmentation
# #     if distance_factor < 0.5:  # Door hai to contrast badhao
# #         lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
# #         l, a, b = cv2.split(lab)
# #         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# #         l = clahe.apply(l)
# #         enhanced = cv2.merge([l, a, b])
# #         crop_processed = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
# #     else:
# #         crop_processed = crop
    
# #     # Model input
# #     img = cv2.resize(crop_processed, (384, 384)).astype(np.float32) / 255.0
# #     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
# #     interpreter.invoke()
# #     pred = interpreter.get_tensor(output_details[0]['index'])[0]
    
# #     # Threshold adjustment based on distance
# #     threshold = 0.45 if distance_factor < 0.5 else 0.5
# #     mask = (np.squeeze(pred) > threshold).astype(np.uint8) * 255
    
# #     # Resize back
# #     mask_resized = cv2.resize(mask, (crop.shape[1], crop.shape[0]))
    
# #     return mask_resized


# # # ==========================
# # # 4️⃣ MAIN LOOP
# # # ==========================
# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(
# #     refine_landmarks=True, 
# #     min_detection_confidence=0.6,  # Thoda kam rakha for better detection
# #     min_tracking_confidence=0.6
# # )

# # cap = cv2.VideoCapture(0)

# # # Previous distance for smoothing
# # prev_dist = None

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret: 
# #         break
# #     frame = cv2.flip(frame, 1)
# #     h, w = frame.shape[:2]
    
# #     # Mediapipe processing
# #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     results = face_mesh.process(rgb_frame)
    
# #     if results.multi_face_landmarks:
# #         landmarks = results.multi_face_landmarks[0].landmark
        
# #         # Calculate distance with smoothing
# #         p1 = np.array([landmarks[468].x, landmarks[468].y])
# #         p2 = np.array([landmarks[473].x, landmarks[473].y])
# #         dist = np.linalg.norm(p1 - p2)
        
# #         # Smooth distance changes
# #         if prev_dist is not None:
# #             dist = 0.7 * dist + 0.3 * prev_dist
# #         prev_dist = dist
        
# #         # 🔥 IMPROVED BOX CALCULATION
# #         # Door se bhi bada box rakho taaki iris achhe se capture ho
# #         # Minimum size 60px (instead of 35px)
# #         base_box = int(dist * w * 1.0)  # 0.85 se 1.0 kiya
# #         b = max(60, min(base_box, 120))  # Minimum 60px, maximum 120px
        
# #         # Distance factor calculate karo (0=far, 1=close)
# #         distance_factor = (b - 60) / (120 - 60)  # Normalize between 0 and 1
        
# #         for idx in [468, 473]:
# #             cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            
# #             # Slicing with boundary safety
# #             y1, y2, x1, x2 = max(0, cy-b), min(h, cy+b), max(0, cx-b), min(w, cx+b)
# #             crop = frame[y1:y2, x1:x2].copy()
            
# #             if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
# #                 # Enhanced mask prediction
# #                 mask = predict_mask(crop, distance_factor)
                
# #                 # Apply improved overlay
# #                 result_eye = apply_png_lens_overlay(crop, mask, lens_img, distance_factor)
                
# #                 # Put back in frame
# #                 if result_eye.shape == frame[y1:y2, x1:x2].shape:
# #                     frame[y1:y2, x1:x2] = result_eye

# #     cv2.imshow("TTDEye Perfect Overlay", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'): 
# #         break

# # cap.release()
# # cv2.destroyAllWindows()

# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import tensorflow as tf

# # # ==========================
# # # 1️⃣ SETUP & MODEL LOADING
# # # ==========================
# # TFLITE_PATH = "iris_pure_float32.tflite"
# # interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
# # interpreter.allocate_tensors()
# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()

# # # Lens Image Load (Make sure trial.png is in the correct folder)
# # LENS_PATH = "images/test.png" 
# # lens_img = cv2.imread(LENS_PATH, cv2.IMREAD_UNCHANGED)

# # # ==========================
# # # 2️⃣ CORE FUNCTIONS (Processing)
# # # ==========================

# # def clean_iris_mask(mask):
# #     """Model ke holes ko bharta hai aur mask ko solid banata hai"""
# #     if mask is None or np.sum(mask) == 0:
# #         return mask
# #     # Holes bharna (Closing)
# #     kernel = np.ones((7, 7), np.uint8)
# #     mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# #     # Mask ko thoda expand karna (Dilation)
# #     mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=1)
# #     # Sirf barest (iris) contour ko rakhna
# #     contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     clean_mask = np.zeros_like(mask_cleaned)
# #     if contours:
# #         largest_contour = max(contours, key=cv2.contourArea)
# #         cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
# #     return cv2.GaussianBlur(clean_mask, (5, 5), 0)

# # def apply_png_lens_overlay(crop, mask, lens_texture):
# #     h, w = crop.shape[:2]
# #     lens_res = cv2.resize(lens_texture, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
# #     # Color aur Alpha alag karein
# #     lens_bgr = lens_res[:, :, :3]
# #     lens_alpha = lens_res[:, :, 3].astype(float) / 255.0 if lens_res.shape[2] == 4 else 1.0

# #     # 1. 🔥 ROBUST CIRCLE FITTING (Door jane wala fix)
# #     # Mask se golaayi nikal kar usse solid daira banaein
# #     solid_mask = np.zeros((h, w), dtype=np.uint8)
# #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
# #     if contours:
# #         # Sabse bare contour ke gird aik perfect circle fit karein
# #         largest = max(contours, key=cv2.contourArea)
# #         (x_c, y_c), radius = cv2.minEnclosingCircle(largest)
# #         # Is circle ko solid white (255) fill karein
# #         cv2.circle(solid_mask, (int(x_c), int(y_c)), int(radius), 255, -1)
# #     else:
# #         return crop # Agar mask na ho to kuch na karein

# #     # 2. Smoothness aur Alpha
# #     mask_norm = cv2.GaussianBlur(solid_mask.astype(float) / 255.0, (5, 5), 0)
# #     final_alpha = np.expand_dims(mask_norm * lens_alpha, axis=-1)

# #     # 3. Final Blending (Ab iris peeche chupa rahega)
# #     result = (crop.astype(float) * (1.0 - final_alpha)) + (lens_bgr.astype(float) * final_alpha)
# #     return result.astype(np.uint8)

# # def predict_mask(crop):
# #     """Prepares image and gets prediction from TFLite"""
# #     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
# #     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
# #     interpreter.invoke()
# #     pred = interpreter.get_tensor(output_details[0]['index'])[0]
# #     mask = (np.squeeze(pred) > 0.5).astype(np.uint8) * 255
# #     return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# # # ==========================
# # # 3️⃣ MAIN REAL-TIME LOOP
# # # ==========================
# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)
# # cap = cv2.VideoCapture(0)

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret: break
# #     frame = cv2.flip(frame, 1)
# #     h, w = frame.shape[:2]
    
# #     results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
# #     if results.multi_face_landmarks:
# #         landmarks = results.multi_face_landmarks[0].landmark
        
# #         # --- TIGHT DYNAMIC BOX ---
# #         # Dono aankhon ke beech distance
# #         p1 = np.array([landmarks[468].x, landmarks[468].y])
# #         p2 = np.array([landmarks[473].x, landmarks[473].y])
# #         dist = np.linalg.norm(p1 - p2)
        
# #         # Box ko tight rakhein (0.75 factor) taaki model ko 'zoom-in' aankh mile
# #         b = int(dist * w * 0.75) 
# #         b = max(35, min(b, 110)) # Scaling safety
        
# #         for idx in [468, 473]:
# #             cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
# #             y1, y2, x1, x2 = max(0, cy-b), min(h, cy+b), max(0, cx-b), min(w, cx+b)
# #             crop = frame[y1:y2, x1:x2].copy()
            
# #             if crop.size > 0:
# #                 raw_mask = predict_mask(crop)
# #                 # Step 1: Repair the mask (Fill holes)
# #                 cleaned_mask = clean_iris_mask(raw_mask)
# #                 # Step 2: Force overlay the lens
# #                 result_eye = apply_png_lens_overlay(crop, cleaned_mask, lens_img)
                
# #                 if result_eye.shape == frame[y1:y2, x1:x2].shape:
# #                     frame[y1:y2, x1:x2] = result_eye

# #     cv2.imshow("Final TTDEye Precise Overlay", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'): break

# # cap.release()
# # cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf

# # ==========================
# # 1️⃣ SETUP & MODEL LOADING
# # ==========================
# # TFLite Model load karein
# TFLITE_PATH = "iris_pure_float32.tflite"
# interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Lens Texture load karein (Make sure path is correct)
# LENS_PATH = "images/1.png" 
# lens_img = cv2.imread(LENS_PATH, cv2.IMREAD_UNCHANGED)

# # Mediapipe Face Mesh Setup
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     refine_landmarks=True, 
#     min_detection_confidence=0.7, 
#     min_tracking_confidence=0.7
# )

# # ==========================
# # 2️⃣ HELPER FUNCTIONS
# # ==========================

# def predict_mask(crop):
#     """UNet model se iris ka mask nikalne ke liye"""
#     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
#     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
#     interpreter.invoke()
#     pred = interpreter.get_tensor(output_details[0]['index'])[0]
#     mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
#     return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# # def apply_hybrid_lens(frame, landmarks, lens_texture):
# #     h, w = frame.shape[:2]
    
# #     # Eyelid points for natural masking
# #     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# #     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# #     eye_configs = [
# #         (468, 159, 145, 469, LEFT_EYE_POINTS), 
# #         (473, 386, 374, 474, RIGHT_EYE_POINTS) 
# #     ]

# #     for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
# #         try:
# #             cx = int(float(landmarks[iris_idx].x) * w)
# #             cy = int(float(landmarks[iris_idx].y) * h)
            
# #             t_y = float(landmarks[top_idx].y) * h
# #             b_y = float(landmarks[bot_idx].y) * h
# #             if abs(t_y - b_y) < (h * 0.012): continue 

# #             ex = int(float(landmarks[edge_idx].x) * w)
# #             ey = int(float(landmarks[edge_idx].y) * h)
# #             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.2)
# #             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
# #             crop = frame[y1:y2, x1:x2].copy()
# #             if crop.size == 0: continue

# #             # --- MASKING LOGIC (Already working) ---
# #             eye_poly = np.array([[(float(landmarks[p].x)*w - x1), (float(landmarks[p].y)*h - y1)] for p in eye_pts], dtype=np.int32)
# #             occlusion_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
# #             cv2.fillPoly(occlusion_mask, [eye_poly], 255)

# #             model_mask = predict_mask(crop) 
# #             ch, cw = crop.shape[:2]
# #             geo_mask = np.zeros((ch, cw), dtype=np.uint8)
# #             cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.9), 255, -1)

# #             combined_mask = cv2.bitwise_or(geo_mask, model_mask)
# #             final_mask = cv2.bitwise_and(combined_mask, occlusion_mask)
# #             final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

# #             # Lens Resize
# #             lens_res = cv2.resize(lens_texture, (x2-x1, y2-y1), interpolation=cv2.INTER_LANCZOS4)
            
# #             if lens_res.shape[2] == 4:
# #                 # --- 🌟 REALISM ENHANCEMENTS START HERE ---
                
# #                 # 1. Specular Highlights Preservation (Asli chamak nikalna)
# #                 # Crop area se wo pixels nikalna jo bohot bright hain (Reflections)
# #                 gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# #                 _, highlight_mask = cv2.threshold(gray_crop, 210, 255, cv2.THRESH_BINARY)
# #                 highlight_mask = cv2.GaussianBlur(highlight_mask, (7, 7), 0)
# #                 highlight_3d = cv2.merge([highlight_mask, highlight_mask, highlight_mask]) / 255.0

# #                 # 2. Top Shadowing (Palkon ka halka saaya)
# #                 # Lens ke upar wale hisse ko thora dark karna
# #                 shadow_map = np.ones((ch, cw), dtype=np.float32)
# #                 shadow_map[:int(ch*0.45), :] = 0.82 # 18% darkness on top
# #                 shadow_map = cv2.GaussianBlur(shadow_map, (15, 15), 0)
# #                 shadow_3d = cv2.merge([shadow_map, shadow_map, shadow_map])

# #                 # 3. Contrast & Color Adjust
# #                 lens_bgr = lens_res[:, :, :3].astype(float)
# #                 lens_bgr = cv2.convertScaleAbs(lens_bgr, alpha=1.1, beta=-10) # Sharpness + Depth

# #                 # 4. Final Advanced Blending
# #                 alpha_map = (final_mask.astype(float) / 255.0) * (lens_res[:,:,3] / 255.0)
# #                 alpha_3d = cv2.merge([alpha_map, alpha_map, alpha_map])
                
# #                 # Lens with shadow applied
# #                 fg = (lens_bgr.astype(float) * shadow_3d) * alpha_3d
# #                 bg = crop.astype(float) * (1.0 - alpha_3d)
                
# #                 # Adding it all together
# #                 blended = cv2.add(fg, bg)
                
# #                 # Re-applying original highlights on top of the lens
# #                 # Isse lens 'plastic' nahi balki 'glassy' lagega
# #                 final_result = cv2.addWeighted(blended, 1.0, crop.astype(float) * highlight_3d, 0.4, 0)
                
# #                 frame[y1:y2, x1:x2] = np.clip(final_result, 0, 255).astype(np.uint8)

# #         except Exception as e:
# #             continue
                
# #     return frame

# def apply_hybrid_lens(frame, landmarks, lens_texture):
#     h, w = frame.shape[:2]
    
#     # 1. Eyelid Landmarks for Natural Occlusion
#     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
#     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

#     eye_configs = [
#         (468, 159, 145, 469, LEFT_EYE_POINTS), 
#         (473, 386, 374, 474, RIGHT_EYE_POINTS) 
#     ]

#     for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
#         try:
#             cx = int(float(landmarks[iris_idx].x) * w)
#             cy = int(float(landmarks[iris_idx].y) * h)
            
#             # Eye Openness Check
#             t_y = float(landmarks[top_idx].y) * h
#             b_y = float(landmarks[bot_idx].y) * h
#             if abs(t_y - b_y) < (h * 0.012): continue 

#             # Radius & Safe Crop
#             ex = int(float(landmarks[edge_idx].x) * w)
#             ey = int(float(landmarks[edge_idx].y) * h)
#             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.25) # Slightly larger for blending
            
#             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
#             crop = frame[y1:y2, x1:x2].copy()
#             if crop.size == 0: continue
#             ch, cw = crop.shape[:2]

#             # 2. 🔥 ADVANCED MASKING (Occlusion + Refinement)
#             eye_poly = np.array([[(float(landmarks[p].x)*w - x1), (float(landmarks[p].y)*h - y1)] for p in eye_pts], dtype=np.int32)
#             occlusion_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.fillPoly(occlusion_mask, [eye_poly], 255)

#             model_mask = predict_mask(crop) 
#             geo_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.95), 255, -1)

#             combined_mask = cv2.bitwise_or(geo_mask, model_mask)
#             final_mask = cv2.bitwise_and(combined_mask, occlusion_mask)
#             # Edge softening for natural limbic ring
#             final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

#             # 3. 🌟 ULTRA-REALISM LAYERS
#             lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
            
#             if lens_res.shape[2] == 4:
#                 # A. Specular Highlights Extraction (Asli chamak bachana)
#                 gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#                 _, highlights = cv2.threshold(gray_crop, 210, 255, cv2.THRESH_BINARY)
#                 highlights = cv2.GaussianBlur(highlights, (9, 9), 0)
#                 highlights_3d = cv2.merge([highlights, highlights, highlights]) / 255.0

#                 # B. Ambient Top Shadow (Palkon ka saaya)
#                 shadow_map = np.ones((ch, cw), dtype=np.float32)
#                 for i in range(int(ch * 0.45)): # Top 45% area
#                     shadow_map[i, :] = 0.8 + (0.2 * (i / (ch * 0.45)))
#                 shadow_3d = cv2.merge([shadow_map, shadow_map, shadow_map])

#                 # C. Final Advanced Blending
#                 lens_bgr = lens_res[:, :, :3].astype(float)
#                 # Contrast adjustment for depth
#                 lens_bgr = cv2.convertScaleAbs(lens_bgr, alpha=1.15, beta=-15)
                
#                 alpha_map = (final_mask.astype(float) / 255.0) * (lens_res[:,:,3] / 255.0)
#                 alpha_3d = cv2.merge([alpha_map, alpha_map, alpha_map])
                
#                 # Apply shadow to lens and blend with background
#                 fg = (lens_bgr * shadow_3d) * alpha_3d
#                 bg = crop.astype(float) * (1.0 - alpha_3d)
#                 blended = cv2.add(fg, bg)
                
#                 # D. Re-Overlay Original Highlights (The "Glassy" look)
#                 # Isse lens 'mat' nahi lagega balki asli shiny cornea lagega
#                 final_result = cv2.addWeighted(blended, 1.0, crop.astype(float) * highlights_3d, 0.5, 0)
                
#                 frame[y1:y2, x1:x2] = np.clip(final_result, 0, 255).astype(np.uint8)

#         except Exception as e:
#             continue
                
#     return frame

# # ==========================
# # 3️⃣ MAIN LOOP
# # ==========================
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
#     frame = cv2.flip(frame, 1) # Mirror effect
    
#     # Process Frame
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)
    
#     if results.multi_face_landmarks:
#         landmarks = results.multi_face_landmarks[0].landmark
#         # Hybrid function call karein
#         frame = apply_hybrid_lens(frame, landmarks, lens_img)

#     cv2.imshow("TTDEye Hybrid Precise Filter", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'): break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# --- 1. Setup ---
interpreter = tf.lite.Interpreter(model_path="iris_pure_float32.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)

lens_img = cv2.imread("images/6.png", cv2.IMREAD_UNCHANGED)

def predict_mask(crop):
    img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    mask = (np.squeeze(pred) > 0.15).astype(np.uint8) * 255
    return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

cap = cv2.VideoCapture(0)
p_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Eye Configurations
        # Left Eye points aur Right Eye points for clipping
        eye_configs = [
            (468, [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]),
            (473, [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398])
        ]

        for iris_idx, eye_pts in eye_configs:
            cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
            # Dynamic radius based on eye distance
            r = int(np.linalg.norm(np.array([landmarks[eye_pts[0]].x*w, landmarks[eye_pts[0]].y*h]) - 
                                   np.array([landmarks[eye_pts[8]].x*w, landmarks[eye_pts[8]].y*h])) / 2.5)
            
            y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
            crop = frame[y1:y2, x1:x2].copy()
            
            if crop.size > 0:
                ch, cw = crop.shape[:2]
                # 1. CALL UNET MODEL
                m_mask = predict_mask(crop)
                
                # 2. MEDIAPIPE CLIPPING (Eyelid mask)
                occ_mask = np.zeros((ch, cw), dtype=np.uint8)
                eye_poly = np.array([[(int(landmarks[p].x * w) - x1), (int(landmarks[p].y * h) - y1)] for p in eye_pts], dtype=np.int32)
                cv2.fillPoly(occ_mask, [eye_poly], 255)
                
                # Final Hybrid Mask
                final_mask = cv2.bitwise_and(m_mask, occ_mask)
                final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

                # 3. LENS OVERLAY
                lens_res = cv2.resize(lens_img, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
                if lens_res.shape[2] == 4:
                    alpha = (lens_res[:, :, 3].astype(float) / 255.0) * (final_mask.astype(float) / 255.0)
                    alpha_3d = cv2.merge([alpha] * 3)
                    
                    fg = lens_res[:, :, :3].astype(float) * alpha_3d
                    bg = crop.astype(float) * (1.0 - alpha_3d)
                    frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)

    # FPS Display
    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
    p_time = c_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("UNet + MediaPipe Local Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()