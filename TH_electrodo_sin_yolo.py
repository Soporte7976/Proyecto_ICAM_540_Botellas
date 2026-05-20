
import os
import cv2
import numpy as np

import time
import threading
import queue       # OPTIMIZACIÓN 1: cola para escritura asíncrona a disco
import os
from pathlib import Path
import torch

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
from CamNavi2 import CamNavi2


# ================= CONFIG =================
SAVE_PATH = "/home/icam-540/Fotos_Botellas_modelo_1"

# Resolución cámara (reducida para mejor rendimiento)
#WIDTH  = 1920
#HEIGHT = 1080
WIDTH  = 3840
HEIGHT = 2160
# Tamaño YOLO (pequeño = procesamiento rápido)
YOLO_SIZE_W = 640 
YOLO_SIZE_H = 480 
YOLO_CONF = 0.5  # Confianza mínima (> 0.5 = más rápido)

MUESTRA_IMAGEN = False
detection_event = threading.Event()

# OPTIMIZACIÓN 2: evento para despertar el loop EXACTO cuando llega un frame
# Evita el time.sleep(0.1) fijo y reduce latencia de respuesta al trigger
frame_event = threading.Event()

_frame_lock = threading.Lock()

print("✅ Modelo TensorRT calentado y listo")

os.makedirs(SAVE_PATH, exist_ok=True)

image_arr = None
resized_2 = None

gain =1
sharpness =5
brightness =10
saturation=10
gamma=5

lista_confi= []
lista_conteo= []
count_unidades = 0
_ultimo_trigger = 0.0
DEBOUNCE_SEG = 0.01  # ignora triggers que lleguen en menos de 500 ms

def lectura_Archivo_Conteo():
    global lista_conteo
    try:
        lista_conteo.clear()
        with open("/home/icam-540/Conteo_Objetos_EL.txt","r", encoding="utf-8") as archivo_C:
            for linea in archivo_C:
                linea = linea.replace('\n','')
                print(linea)
                lista_conteo.append(linea)
    except Exception as ex:
        print(f"Error lectura Archivo CONTEO {ex}")


def lectura_Confisistema():
    global lista_confi
    try:
        lista_confi.clear()
        with open("/home/icam-540/CONFISISTEMA_EL.txt","r", encoding="utf-8") as archivo:
            for linea in archivo:
                linea = linea.replace('\n','')
                print(linea)
                lista_confi.append(linea)
    except Exception as ex:
        print(f"Error lectura CONFISISTEMA_EL {ex}")

# OPTIMIZACIÓN 1: Cola de escritura asíncrona a disco
# El loop principal ya no se bloquea esperando I/O del archivo.
# Las escrituras se procesan en un hilo separado de fondo (daemon).
_file_queue = queue.Queue(maxsize=10)

def _writer_thread():
    """Hilo daemon: consume la cola y escribe Conteo_Objetos_EL.txt sin bloquear el loop."""
    while True:
        linea, valor = _file_queue.get()   # espera hasta que haya algo
        try:
            with open("/home/icam-540/Conteo_Objetos_EL.txt", "r", encoding="utf-8") as archivo:
                lineas = archivo.readlines()
            while len(lineas) <= linea:
                lineas.append("\n")
            lineas[linea] = str(valor) + "\n"
            with open("/home/icam-540/Conteo_Objetos_EL.txt", "w", encoding="utf-8") as archivo:
                archivo.writelines(lineas)
        except Exception as e:
            print(f"Actualizar archivo {e}")

# Inicia el hilo de escritura como daemon (se cierra solo al terminar el programa)
threading.Thread(target=_writer_thread, daemon=True, name="FileWriter").start()

def actualizar_linea_archivo(linea, valor):
    """Encola la escritura en lugar de escribir directamente — no bloquea el loop."""
    try:
        _file_queue.put_nowait((linea, valor))
    except queue.Full:
        print("?? Cola escritura llena  escritura descartada")

# ---------- Convert GST buffer to OpenCV ----------
def gst_to_opencv(sample):
    buf = sample.get_buffer()
    buffer = buf.extract_dup(0, buf.get_size())
    arr = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ---------- IMAGE CALLBACK (se llama por TRIGGER HARDWARE) ----------
def new_image_handler(sample):
    global image_arr
    global MUESTRA_IMAGEN
    global bandera_Yolo
    global count_unidades
    global _ultimo_trigger
    global contador_imagenes
    if sample is None:
        return
    ahora = time.time()
    if ahora - _ultimo_trigger < DEBOUNCE_SEG:
        print(f"⚠️ Trigger ignorado (rebote) Δt={ahora - _ultimo_trigger:.3f}s")
        return
    _ultimo_trigger = ahora
    img = gst_to_opencv(sample)
    with _frame_lock:
        image_arr = img

    count_unidades += 1
    print(f"✅ Contador: {count_unidades}")
    if count_unidades == 1:
        bandera_Yolo = False
        count_unidades = 0
    else:
        bandera_Yolo = True
        image_arr = None
    if contador_imagenes == 350:
        contador_imagenes = 0
    # OPTIMIZACIÓN 2: notifica al loop principal que llegó un frame nuevo.
    # El loop deja de dormir inmediatamente en lugar de esperar el sleep fijo.
    frame_event.set()

def save_detection(frame, nombre_img):
    """Guarda imagen de detección"""
    try:
        cv2.imwrite(str(SAVE_PATH +"/"+ nombre_img), frame)
        print(f"✅ Detección guardada: {SAVE_PATH+  nombre_img}")
        detection_event.set()
    except Exception as e:
        print(f"❌ Error al guardar: {e}")

    
if __name__ == '__main__':
    lectura_Confisistema()
    lectura_Archivo_Conteo()
    bandera_Yolo = False
    count_rechazo = 0
   
    count_bueno = int(lista_conteo[0])
    count_malo= int(lista_conteo[1])
    linea_cero = 0
    linea_uno = 1
    try:
        cn2 = CamNavi2.CamNavi2()
    except:
        cn2 = CamNavi2()

    # Enumerar cámaras
    camera_dict = cn2.enum_camera_list()
    print("Cámaras detectadas:", camera_dict)

    camera = cn2.get_device_by_name('iCam500')  # iCAM-540 usa este driver
    icam_color = int(cn2.advcam_query_fw_sku(camera))

    # ---------- PIPELINE ----------
    pipe_params = {
        "acq_mode": 2,
        "width": WIDTH,
        "height": HEIGHT,
        "enable_infer": 0
    }

    if icam_color == 1:
        pipe_params["format"] = "YUY2"

    cn2.advcam_config_pipeline(camera, **pipe_params)
    cn2.advcam_open(camera, -1)
    # Setting do0 parameters
   #  camera.dio.do0.op_mode = 0 # DO op mode: user output
   #  camera.dio.do0.reverse = 0
    camera.dio.do0.user_output = 0 # DO low, DI high
    print("DO lOW " + str(camera.dio.do0.user_output))
   


    # ---------- REGISTER CALLBACK ----------
    cn2.advcam_register_new_image_handler(camera, new_image_handler)


    camera.hw_trigger_delay = 0
    print("Delay " + str(camera.hw_trigger_delay))

    camera.lighting.selector = 3

    camera.lighting.gain = 100
     #camera.lighting.selector = int(lista_confi[0])

     #camera.lighting.gain = int(lista_confi[1])
     #camera.image.saturation = int(lista_confi[2])
     #camera.image.gamma = int(lista_confi[3])

    #cn2.advcam_set_img_sharpness(camera, int(lista_confi[4]))
    ##cn2.advcam_set_img_brightness(camera,  int(lista_confi[5]))
    #cn2.advcam_set_img_gain(camera, int(lista_confi[6]))
    
    camera.image.saturation = 208
    camera.image.gamma = 60
        
    cn2.advcam_set_img_sharpness(camera, 100)
    cn2.advcam_set_img_brightness(camera, 20)
    cn2.advcam_set_img_gain(camera, 6)
    camera.focus.pos_zero()
    print("Exposición actual:", camera.image.exposure_time)
    camera.image.exposure_time = int(10)
    print("Exposición nueva:", camera.image.exposure_time)
    
    camera.focus.distance = 60
    #camera.focus.distance = int(lista_confi[7])
    #contador_imagenes = int(lista_confi[8])
    contador_imagenes = 0
    print("lens motor posistion: ", camera.focus.position())
    i = 0
    while i < 6:
            camera.focus.direction = 1 # lens focusing motor backward
            try:
                camera.focus.distance = 100
                print("lens motor posistion: ", camera.focus.position())
                time.sleep(0.1) 
                i+=1
                print("valor ", i)
            except ValueError:
                print("lens position out of index")
    camera.focus.direction = 1
    camera.focus.distance = 10
     
     #print("lens motor posistion: ", camera.focus.position())
    # ---------- START STREAM ----------
   
    cn2.advcam_play(camera)

    print("✅ iCAM-540 listo. Esperando trigger hardware en PIN 10...")
    ultimo_frame = None
    frame_yolo = None
    resized = None
    try:
        while True:
            # OPTIMIZACIÓN 2: espera hasta que llegue un frame nuevo (máx 500ms)
                # Reemplaza el time.sleep(0.1) fijo — el loop despierta exacto con el trigger
            frame_event.wait(timeout=0.5)
            frame_event.clear()


            with _frame_lock:
                 frame_local = image_arr
                 image_arr = None

            if frame_local  is not None:
               

                try:
                 resized = cv2.resize(frame_local, (YOLO_SIZE_W, YOLO_SIZE_H))
                except Exception:
                 pass
                
                contador_imagenes+=1
                nombre_ig = "foto_" +str(contador_imagenes) + ".png"
                save_detection(resized, nombre_ig)
                cv2.imshow("Vista Camara",resized)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break
                elif key == ord('-'): 
                    cv2.destroyAllWindows()
                    with _frame_lock:
                        image_arr = None
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
                elif key == ord('n') or key == ord('N'):  # Exposición +10 ms
                    try:
                        nuevo = max(10, min(10000, int(camera.image.exposure_time + 1)))
                        camera.image.exposure_time = nuevo
                        print("Exposición Nueva:", camera.image.exposure_time)
                    except:
                        pass
                elif key == ord('c') or key == ord('C'):  # GAIN AUMENTAR
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
                elif key == ord('w') or key == ord('W'):  # SHARPNESS DISMINUIR
                    try:

                        sharpness_a = cn2.advcam_get_img_sharpness(camera)
                        sharpness_a = sharpness_a - sharpness
                        cn2.advcam_set_img_sharpness(camera, sharpness_a)
                        print("sharpness disminuir: ", cn2.advcam_get_img_sharpness(camera))
                    except Exception as e:
                        print(f"❌ Error sharpness d: {e}")
                    pass
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
                elif key == ord('K') or key == ord('k'):  # SATURATION AUMENTAR
                    try:

                        saturation_a = camera.image.saturation
                        saturation_a = saturation_a + saturation
                        camera.image.saturation =  saturation_a
                        print("SATURACION aumento: ", camera.image.saturation)
                    except Exception as e:
                        print(f"❌ Error SATURACION  a: {e}")
                        pass
                elif key == ord('J') or key == ord('j'):  # SATURATION DISMINUIR
                    try:

                        saturation_a = camera.image.saturation
                        saturation_a = saturation_a - saturation
                        camera.image.saturation =  saturation_a
                        print("SATURACION disminuir: ", camera.image.saturation)
                    except Exception as e:
                        print(f"❌ Error SATURACION d: {e}")
                        pass 
                elif key == ord('U') or key == ord('u'):  # GAMMA AUMENTAR
                    try:

                        gamma_a = camera.image.gamma
                        gamma_a = gamma_a + gamma
                        camera.image.gamma =  gamma_a
                        print("gamma aumento: ", camera.image.gamma)
                    except Exception as e:
                        print(f"❌ Error gamma  a: {e}")
                        pass
                elif key == ord('l') or key == ord('L'):  # GAMMA DISMINUIR
                    try:

                        gamma_a = camera.image.gamma
                        gamma_a = gamma_a - gamma
                        camera.image.gamma =  gamma_a
                        print("gamma disminuyo: ", camera.image.gamma)
                    except Exception as e:
                        print(f"❌ Error gamma d: {e}")
                        pass
            else:
                count_rechazo+=1
                if count_rechazo == 2 and  str(camera.dio.do0.user_output) == "1":
                    camera.dio.do0.user_output = 0
                    salida  = str(camera.dio.do0.user_output)
                    print("DO Low " + salida)

                if frame_yolo is not None:
                    ultimo_frame = frame_yolo.copy()

                if resized is not None  and frame_yolo is None:
                    ultimo_frame = resized.copy()

                if ultimo_frame is not None :
                    cv2.imshow("Vista Camara",ultimo_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('-'): 
                    cv2.destroyAllWindows()
                    with _frame_lock:
                        image_arr = None
                elif key == ord('t'):  
                    count_rechazo = 0
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
                elif key == ord('n') or key == ord('N'):  # Exposición +10 ms
                    try:
                        nuevo = max(10, min(10000, int(camera.image.exposure_time + 1)))
                        camera.image.exposure_time = nuevo
                        print("Exposición Nueva:", camera.image.exposure_time)
                    except:
                        pass
                elif key == ord('c') or key == ord('C'):  # GAIN AUMENTAR
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
                elif key == ord('w') or key == ord('W'):  # SHARPNESS DISMINUIR
                    try:

                        sharpness_a = cn2.advcam_get_img_sharpness(camera)
                        sharpness_a = sharpness_a - sharpness
                        cn2.advcam_set_img_sharpness(camera, sharpness_a)
                        print("sharpness disminuir: ", cn2.advcam_get_img_sharpness(camera))
                    except Exception as e:
                        print(f"❌ Error sharpness d: {e}")
                    pass
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
                elif key == ord('K') or key == ord('k'):  # SATURATION AUMENTAR
                    try:

                        saturation_a = camera.image.saturation
                        saturation_a = saturation_a + saturation
                        camera.image.saturation =  saturation_a
                        print("SATURACION aumento: ", camera.image.saturation)
                    except Exception as e:
                        print(f"❌ Error SATURACION  a: {e}")
                        pass
                elif key == ord('J') or key == ord('j'):  # SATURATION DISMINUIR
                    try:

                        saturation_a = camera.image.saturation
                        saturation_a = saturation_a - saturation
                        camera.image.saturation =  saturation_a
                        print("SATURACION disminuir: ", camera.image.saturation)
                    except Exception as e:
                        print(f"❌ Error SATURACION d: {e}")
                        pass 
                elif key == ord('U') or key == ord('u'):  # GAMMA AUMENTAR
                    try:

                        gamma_a = camera.image.gamma
                        gamma_a = gamma_a + gamma
                        camera.image.gamma =  gamma_a
                        print("gamma aumento: ", camera.image.gamma)
                    except Exception as e:
                        print(f"❌ Error gamma  a: {e}")
                        pass
                elif key == ord('l') or key == ord('L'):  # SATURATION DISMINUIR
                    try:

                        gamma_a = camera.image.gamma
                        gamma_a = gamma_a - gamma
                        camera.image.gamma =  gamma_a
                        print("gamma disminuyo: ", camera.image.gamma)
                    except Exception as e:
                        print(f"❌ Error gamma d: {e}")
                        pass
    except KeyboardInterrupt:
        pass

    # ---------- CLEANUP ----------
    cv2.destroyAllWindows()
    cn2.advcam_register_new_image_handler(camera, None)
    cn2.advcam_close(camera)
