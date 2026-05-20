#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SCRIPT OPTIMIZADO PARA iCAM-540 + YOLO
Basado en Video_10_YOlO.py con mejoras de rendimiento
"""

import platform

import cv2
import time
import threading
import os
import torch
import numpy as np
from pathlib import Path
from ultralytics.utils.ops import Profile

# ================= CONFIGURACIÓN =================
print("CUDA:", torch.cuda.is_available())
print("CUDA conteo:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No hay CUDA disponible")
# ===== Patch compatibilidad torchvision con torch NVIDIA Jetson (CUDA 12.6) =====

# Patch 1: evita error al registrar operadores meta de torchvision
import torch._library.fake_impl as _fake_impl_module
_orig_fake_register = _fake_impl_module.FakeImplHolder.register
def _patched_fake_register(self, func, source):
    try:
        return _orig_fake_register(self, func, source)
    except RuntimeError:
        return None
_fake_impl_module.FakeImplHolder.register = _patched_fake_register

# Patch 2: reemplaza torchvision.ops.nms con implementacion pura PyTorch
#          (el .so de torchvision no es compatible con el build NVIDIA)
import torchvision.ops as _tv_ops
def _nms_puro(boxes, scores, iou_threshold):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = boxes[order[1:], 0].clamp(min=boxes[i, 0])
        yy1 = boxes[order[1:], 1].clamp(min=boxes[i, 1])
        xx2 = boxes[order[1:], 2].clamp(max=boxes[i, 2])
        yy2 = boxes[order[1:], 3].clamp(max=boxes[i, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        areas = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + areas - inter)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
_tv_ops.nms = _nms_puro

# ============================================================================

from ultralytics import YOLO
PT_PATH     = "/home/icam-540/Proyectos/Proyecto_ICAM_540_Botellas/best_botella_2.pt"
#ENGINE_PATH = "/home/icam-540/Proyectos/PROYECTO_ICAM_540_BOTELLAS/best_botella_1.engine"
# ✅ Rutas Compatible Windows/Linux
SAVE_PATH = "/home/icam-540/capturas/captura_FALLA.jpg"

# Resolución cámara (reducida para mejor rendimiento)
WIDTH  = 3840
HEIGHT = 2160
#WIDTH  = 1920
#HEIGHT = 1080
# Tamaño YOLO (pequeño = procesamiento rápido)
YOLO_SIZE_W = 640
YOLO_SIZE_H = 480
YOLO_CONF = 0.7 # Confianza mínima (> 0.5 = más rápido)



# ================= VARIABLES GLOBALES =================
# Exporta a TensorRT Engine solo si no existe; de lo contrario carga directo
if not Path(PT_PATH).exists():
    print(f"[YOLO] best.engine no encontrado — exportando desde {PT_PATH} ...")
    _tmp = YOLO(PT_PATH)
    _tmp.export(format="engine", device=0, half=True)
    del _tmp

model = YOLO(PT_PATH)

# OPTIMIZACIÓN 1: warmup — elimina el spike de latencia en las primeras inferencias
_dummy = np.zeros((YOLO_SIZE_H, YOLO_SIZE_W, 3), dtype=np.uint8)
model(_dummy, verbose=False, half=True)
print("✅ Modelo TensorRT calentado y listo")

latest_frame = None
detection_event = threading.Event()
# OPTIMIZACIÓN 2: evento para despertar el loop exacto cuando llega nuevo frame
frame_event = threading.Event()
# OPTIMIZACIÓN 4: lock para proteger latest_frame de race condition callback/loop
_frame_lock = threading.Lock()

# Contadores FPS
cam_fps = 0
yolo_fps = 0
cam_count = 0
yolo_count = 0
last_fps_time = time.time()

icam_color = 0

gain =1
sharpness =5
brightness =10
gamma = 5
saturation = 10
resized_2 = None
# ================= FUNCIONES =================
frame_yolo = None
resized = None


def gst_to_opencv(sample):
    """Convierte formato SDK (GST) a formato OpenCV"""
    buf = sample.get_buffer()
    data = buf.extract_dup(0, buf.get_size())
    arr = np.frombuffer(data, dtype=np.uint8)

    if icam_color == 1:
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def new_image_handler(sample):
    """
    Callback de CÁMARA (debe ser LIVIANO, sin YOLO aquí)
    Solo captura frames, procesamiento en main loop
    """
    global latest_frame, cam_count

    if sample is None:
        return

    try:
        frame = gst_to_opencv(sample)
        with _frame_lock:
            latest_frame = frame
        frame_event.set()
        cam_count += 1

    except Exception as e:
        print(f"❌ Error en callback cámara: {e}")


def save_detection(frame, class_name="MALO"):
    """Guarda imagen de detección"""
    try:
        cv2.imwrite(str(SAVE_PATH), frame)
        print(f"✅ Detección guardada: {SAVE_PATH}")
        detection_event.set()
    except Exception as e:
        print(f"❌ Error al guardar: {e}")

seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
# ================= MAIN =================
if __name__ == "__main__":
    print("🚀 Iniciando iCAM-540 + YOLO con threading...")

    count_rechazo = 0
    linea_cero = 0
    linea_uno = 1
    count_M = 0
    count_B = 0
    try:
        from CamNavi2 import CamNavi2
        try:
            cn2 = CamNavi2.CamNavi2()
        except:
            cn2 = CamNavi2()
    except ImportError:
        print("❌ Error: No se encontró CamNavi2. Instala el SDK de iCAM-540")
        exit(1)

    try:
        # Obtener cámara
        camera_list = cn2.enum_camera_list()
        if not camera_list:
            print("❌ No se encontró cámara iCAM")
            exit(1)
        
        camera = cn2.get_device_by_name("iCam500")  # O "iCam540" si es 540
        print(f"✅ Cámara detectada: {camera}")

        # Verificar color/mono
        icam_color = int(cn2.advcam_query_fw_sku(camera))
        print(f"📷 Modo: {'Color' if icam_color == 1 else 'Mono'}")

        # ========== CONFIGURAR PIPELINE ==========
        pipe_params = {
            "acq_mode": 0,
            "width": WIDTH,
            "height": HEIGHT,
            "enable_infer": 0  # IMPORTANTE: sin inferencia en cámara
        }

        if icam_color == 1:
            pipe_params["format"] = "YUY2"

        cn2.advcam_config_pipeline(camera, **pipe_params)
        cn2.advcam_open(camera, -1)
        cn2.advcam_register_new_image_handler(camera, new_image_handler)
        camera.dio.do0.op_mode = 0
        camera.dio.do0.user_output = 0 # DO low, DI high
        print("DO lOW " + str(camera.dio.do0.user_output))
        camera.dio.do0.reverse = 0
        # ========== CONFIGURACIÓN ÓPTICA ==========
        camera.lighting.selector = 3
        camera.lighting.gain = 50
        camera.image.saturation = 119
        camera.image.gamma = 45
        
        cn2.advcam_set_img_sharpness(camera, 15)
        cn2.advcam_set_img_brightness(camera, 40)
        cn2.advcam_set_img_gain(camera, 8)
        camera.focus.pos_zero()
        print("Exposición actual:", camera.image.exposure_time)
        camera.image.exposure_time = int(10)
        print("Exposición nueva:", camera.image.exposure_time)

        #camera.set_acq_frame_rate(TARGET_FPS)

        # ========== INICIAR CAPTURA ==========
        
        camera.focus.pos_zero()
        time.sleep(0.5)
        camera.focus.distance = 10

        i = 0
        bandera_cv = False
        while i < 7:
            
            camera.focus.direction = 1 # lens focusing motor backward
            try:
                camera.focus.distance = 100
                print("lens motor posistion: ", camera.focus.position())
                i+=1
                print("valor ", i)
            except ValueError:
                print("lens position out of index")
        
        cn2.advcam_play(camera)
        print("▶️  Sistema iniciado. Presiona ESC para salir")
        print(f"   📊 Resolución cámara: {WIDTH}x{HEIGHT}")
        print(f"   🧠 Tamaño YOLO: {YOLO_SIZE_W}x{YOLO_SIZE_H}")
        print(f"   ⚙️  Confianza YOLO: {YOLO_CONF}")
        print()
        # ================= LOOP PRINCIPAL =================
        while True:
            with _frame_lock:  # OPTIMIZACIÓN 4: lectura atómica del frame
                frame_actual = latest_frame

            if frame_actual is not None:
                count_rechazo+=1
                if count_rechazo == 2 and  str(camera.dio.do0.user_output) == "1":
                    camera.dio.do0.user_output = 0
                    salida  = str(camera.dio.do0.user_output)
                    print("DO Low " + salida)

                resized = cv2.resize(frame_actual, (YOLO_SIZE_W, YOLO_SIZE_H))

                results = model(resized, verbose=False, conf=YOLO_CONF, half=True)

                frame_yolo = results[0].plot()

                for i, cls_id in enumerate(results[0].boxes.cls.tolist()):
                    class_name = results[0].names[int(cls_id)]
                    cof = results[0].boxes.conf[i].item()
                    if class_name == "M":
                        #print(f"🎯 Objeto detectado: {class_name} - Confianza {cof:.2f}")
                        count_M += 1
                        count_rechazo = 0
                        camera.dio.do0.user_output = 1
                        salida = str(camera.dio.do0.user_output)
                        print("DO high " + salida)
                        x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
                        h, w = frame_yolo.shape[:2]

                        banner_h = 80

                            # Crear copia para overlay
                        overlay = frame_yolo.copy()

                            # Dibujar rectángulo en overlay
                        cv2.rectangle(
                                overlay,
                                (0, 0),
                                (w, banner_h),
                                (0, 0, 255),
                                -1
                        )

                            # ?? Transparencia
                        alpha = 0.2  # 0.0 = invisible | 1.0 = sólido

                            # Mezclar overlay con imagen original
                        cv2.addWeighted(
                                overlay,
                                alpha,
                                frame_yolo,
                                1 - alpha,
                                0,
                                frame_yolo
                            )

                            # Texto "DEFECTO" centrado en el banner
                        texto = "DEFECTO"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.8
                        thickness = 3
                        (tw, th), _ = cv2.getTextSize(texto, font, font_scale, thickness)
                        tx = (w - tw) // 2
                        ty = (banner_h + th) // 2
                        cv2.putText(frame_yolo, texto, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    if class_name == "B":
                       # print(f"🎯 Objeto detectado: {class_name} - Confianza {cof:.2f}")
                        count_B += 1
                        x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
                        

                #if platform.system() == 'Linux' and p not in windows:
                 #   windows.append(p)
                if bandera_cv == False:
                    cv2.namedWindow("Vista Camara", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow("Vista Camara", WIDTH, HEIGHT)
                    bandera_cv = True

                frame_yolo_resized = cv2.resize(frame_yolo, (WIDTH, HEIGHT))
                cv2.imshow("Vista Camara", frame_yolo_resized)
                #print(f"B :{count_B}, M: {count_M}")

            # -------- CONTROLES TECLADO --------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n⏹️  Deteniendo...")
                break
            elif key == ord('a') or key == ord('A'):  # Enfoque RETROCEDE
                try:
                    camera.focus.direction = 0
                    camera.focus.distance = 5
                    print("lens motor Retrocede posistion: ", camera.focus.position())
                except:
                    pass
            elif key == ord('b') or key == ord('B'):  # Enfoque ADELANTA
                try:
                    camera.focus.direction = 1
                    camera.focus.distance = 5
                    print("lens motor Adelante posistion: ", camera.focus.position())
                except:
                    pass

                #GAIN
            elif key == ord('n') or key == ord('N'):  # GAIN AUMENTAR
                try:

                    gain_a = cn2.advcam_get_img_gain(camera)
                    gain_a = gain_a + gain
                    cn2.advcam_set_img_gain(camera, gain_a)
                    print("gain aumento: ", cn2.advcam_get_img_gain(camera))
                except Exception as e:
                    print(f"❌ Error gain  a: {e}")
                    pass
            elif key == ord('m') or key == ord('M'):  # GAIN DISMINUIR
                try:

                    gain_a = cn2.advcam_get_img_gain(camera)
                    gain_a = gain_a - gain
                    cn2.advcam_set_img_gain(camera, gain_a)
                    print("gain disminuir: ", cn2.advcam_get_img_gain(camera))
                except Exception as e:
                    print(f"❌ Error gain d: {e}")
                    pass

                #SHARPNESS
            elif key == ord('v') or key == ord('V'):  # SHARPNESS AUMENTAR
                try:

                    sharpness_a = cn2.advcam_get_img_sharpness(camera)
                    sharpness_a = sharpness_a + sharpness
                    cn2.advcam_set_img_sharpness(camera, sharpness_a)
                    print("sharpness aumento: ", cn2.advcam_get_img_sharpness(camera))
                except Exception as e:
                    print(f"❌ Error sharpness  a: {e}")
                    pass
            elif key == ord('c') or key == ord('C'):  # SHARPNESS DISMINUIR
                try:

                    sharpness_a = cn2.advcam_get_img_sharpness(camera)
                    sharpness_a = sharpness_a - sharpness
                    cn2.advcam_set_img_sharpness(camera, sharpness_a)
                    print("sharpness disminuir: ", cn2.advcam_get_img_sharpness(camera))
                except Exception as e:
                    print(f"❌ Error sharpness d: {e}")
                    pass

                #BRIGTHNESS
            elif key == ord('x') or key == ord('X'):  # BRIGTHNESS AUMENTAR
                try:

                    brightness_a = cn2.advcam_get_img_brightness(camera)
                    brightness_a = brightness_a + brightness
                    cn2.advcam_set_img_brightness(camera, brightness_a)
                    print("brightness aumento: ", cn2.advcam_get_img_brightness(camera))
                except Exception as e:
                    print(f"❌ Error brightness  a: {e}")
                    pass
            elif key == ord('z') or key == ord('Z'):  # BRIGTHNESS DISMINUIR
                try:

                    brightness_a = cn2.advcam_get_img_brightness(camera)
                    brightness_a = brightness_a - brightness
                    cn2.advcam_set_img_brightness(camera, brightness_a)
                    print("brightness disminuir: ", cn2.advcam_get_img_brightness(camera))
                except Exception as e:
                    print(f"❌ Error brightness d: {e}")
                    pass    
            elif key == ord('s'):  # Captura manual
                if latest_frame is not None:
                    save_detection(frame_yolo, "Manual")
            elif key == ord('t'):  
                    
                    camera.dio.do0.user_output = 1
                    salida  = str(camera.dio.do0.user_output)
                    print("DO high " + salida)
                    #camera.dio.do0.user_output = 1 # DO high, DI low
                    level =  camera.dio.di0.level
                    print(level)
            elif key == ord('r'):
                    camera.dio.do0.user_output = 0
                    salida  = str(camera.dio.do0.user_output)
                    print("DO low " + salida)
                    #camera.dio.do0.user_output = 1 # DO high, DI low
                    level =  camera.dio.di0.level
                    print(level)
            elif key == ord('u'):
                try:

                    gamma_a =  camera.image.gamma 
                    gamma_a = gamma_a + gamma
                    camera.image.gamma = gamma_a
                    print("gamma aumento: ", camera.image.gamma)
                except Exception as e:
                    print(f"❌ Error gamma  a: {e}")
                    pass
            elif key == ord('l'):
                try:
                    gamma_a =  camera.image.gamma
                    gamma_a = gamma_a - gamma
                    camera.image.gamma = gamma_a
                    print("gamma disminuir: ", camera.image.gamma)
                except Exception as e:
                    print(f"❌ Error gamma d: {e}")
                    pass
            elif key == ord('k'):
                try:
                    saturation_a =  camera.image.saturation
                    saturation_a = saturation_a + saturation
                    camera.image.saturation = saturation_a
                    print("saturation aumento: ", camera.image.saturation)
                except Exception as e:
                    print(f"❌ Error saturation  a: {e}")
                    pass
            elif key == ord('j'):
                try:
                    saturation_a =  camera.image.saturation
                    saturation_a = saturation_a - saturation
                    camera.image.saturation = saturation_a
                    print("saturation disminuir: ", camera.image.saturation)
                except Exception as e:
                    print(f"❌ Error saturation d: {e}")
                    pass

            # OPTIMIZACIÓN 2: duerme hasta que llegue nuevo frame o pasen 50ms
            # evita quemar CPU en el spin-loop cuando bandera_Yolo == True
            frame_event.wait(timeout=0.05)
            frame_event.clear()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ========== LIMPIEZA ==========
        print("\n🧹 Limpiando recursos...")
        try:
            camera.lighting.selector = 0
            camera.lighting.gain = 0
            cn2.advcam_register_new_image_handler(camera, None)
            cn2.advcam_close(camera)
        except:
            pass
        cv2.destroyAllWindows()
        print("✅ Finalizado")
