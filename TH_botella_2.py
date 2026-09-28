
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


from ultralytics import YOLO
from CamNavi2 import CamNavi2


# ================= CONFIG =================
SAVE_PATH = "/home/icam-540/best_botella_2_origen"
SAVE_PATH_YOLO = "/home/icam-540/best_botella_2_yolo"

PT_PATH     = "/home/icam-540/Proyectos/Proyecto_ICAM_540_Botellas/best_botella_2.pt"
#ENGINE_PATH = "/home/icam-540/Proyectos/Proyecto_ICAM_540_Botellas/best_botella_2.engine"

# Resolución cámara (reducida para mejor rendimiento)
#WIDTH  = 1920
#HEIGHT = 1080
WIDTH  = 3840
HEIGHT = 2160
# Tamaño YOLO (pequeño = procesamiento rápido)
YOLO_SIZE_W = 640 
YOLO_SIZE_H = 480 
YOLO_CONF = 0.7  # Confianza mínima (> 0.5 = más rápido)

MUESTRA_IMAGEN = False
detection_event = threading.Event()

# OPTIMIZACIÓN 2: evento para despertar el loop EXACTO cuando llega un frame
# Evita el time.sleep(0.1) fijo y reduce latencia de respuesta al trigger
frame_event = threading.Event()

_frame_lock = threading.Lock()
# =========================================
#  Exporta a TensorRT Engine solo si no existe; de lo contrario carga directo
#  OPTIMIZACIÓN 3: half=True genera engine FP16 → ~30-50% más rápido en Jetson Orin
#if not Path(ENGINE_PATH).exists():
 #   print(f"[YOLO] best.engine no encontrado — exportando desde {PT_PATH} ...")
 #   _tmp = YOLO(PT_PATH)
 #   _tmp.export(format="engine", device=0, half=True)  # FP16 para Jetson GPU
 #   del _tmp

model = YOLO(PT_PATH )

# OPTIMIZACIÓN 4: Warmup del modelo — hace 1 inferencia dummy al arrancar
# La primera inferencia real de TensorRT inicializa contextos CUDA internos (~500ms).
# Con el warmup ese costo ocurre aquí y no en el primer trigger de producción.
_dummy = np.zeros((YOLO_SIZE_H, YOLO_SIZE_W, 3), dtype=np.uint8)
model(_dummy, verbose=False, half=True)
print("✅ Modelo TensorRT calentado y listo")

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH_YOLO, exist_ok=True)
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
    # print(f"✅ Contador: {count_unidades}")
    if count_unidades == 1:
        bandera_Yolo = False
        count_unidades = 0
    else:
        bandera_Yolo = True
        image_arr = None
    
    if contador_imagenes == 1000:
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

def save_detection_yolo(frame, nombre_img):
    """Guarda imagen de detección"""
    try:
        cv2.imwrite(str(SAVE_PATH_YOLO +"/"+ nombre_img), frame)
        print(f"✅ Detección guardada: {SAVE_PATH_YOLO+  nombre_img}")
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

   
     #camera.lighting.selector = int(lista_confi[0])

     #camera.lighting.gain = int(lista_confi[1])
     #camera.image.saturation = int(lista_confi[2])
     #camera.image.gamma = int(lista_confi[3])

    #cn2.advcam_set_img_sharpness(camera, int(lista_confi[4]))
    ##cn2.advcam_set_img_brightness(camera,  int(lista_confi[5]))
    #cn2.advcam_set_img_gain(camera, int(lista_confi[6]))
    camera.lighting.selector = 3
    camera.lighting.gain =    50                                                                                          
    camera.image.saturation = 119
    camera.image.gamma = 40
        
    cn2.advcam_set_img_sharpness(camera, 15)
    cn2.advcam_set_img_brightness(camera, 40)
    cn2.advcam_set_img_gain(camera, 8)
    camera.focus.pos_zero()
    print("Exposición actual:", camera.image.exposure_time)
    camera.image.exposure_time = int(10)
    print("Exposición nueva:", camera.image.exposure_time)
    
    camera.focus.distance = 10
    #camera.focus.distance = int(lista_confi[7])
    #contador_imagenes = int(lista_confi[8])
    contador_imagenes = 0
    print("lens motor posistion: ", camera.focus.position())
    i = 0
    while i < 7:
            camera.focus.direction = 1 # lens focusing motor backward
            try:
                camera.focus.distance = 100
                print("lens motor posistion: ", camera.focus.position())
                time.sleep(0.1) 
                i+=1
                print("valor ", i)
            except ValueError:
                print("lens position out of index")
     #camera.focus.distance = 10
     #camera.focus.direction = 1
     #print("lens motor posistion: ", camera.focus.position())
    # ---------- START STREAM ----------
    cn2.advcam_play(camera)

    print("✅ iCAM-540 listo. Esperando trigger hardware en PIN 10...")
    ultimo_frame = None
    frame_yolo = None
    resized = None
    bandera_cv = False
    try:
        while True:
            # OPTIMIZACIÓN 2: espera hasta que llegue un frame nuevo (máx 500ms)
                # Reemplaza el time.sleep(0.1) fijo — el loop despierta exacto con el trigger
            frame_event.wait(timeout=0.1)
            frame_event.clear()


            with _frame_lock:
                 frame_local = image_arr
                 image_arr = None

            if frame_local  is not None:
               

                try:
                 resized = cv2.resize(frame_local, (YOLO_SIZE_W, YOLO_SIZE_H))
                except Exception:
                 pass
                

                if bandera_Yolo == False:
                    # OPTIMIZACIÓN 3: half=True activa inferencia FP16 en cada frame
                    # Aprovecha el engine compilado con half=True → menor latencia por inferencia
                    results = model(resized, verbose=False, conf=YOLO_CONF, half=True)

                    frame_yolo = results[0].plot()
                    bandera_Yolo = True
                    resized_2 = resized.copy() 
                    contador_imagenes+=1
                    nombre_ig = "foto_" +str(contador_imagenes) + ".png"
                    save_detection(resized, nombre_ig)
                    save_detection_yolo(frame_yolo, nombre_ig)

                    for i,cls_id in enumerate(results[0].boxes.cls.tolist()):
                        class_name = results[0].names[int(cls_id)]
                        if class_name == "M":
                            count_malo+=1
                            count_rechazo = 0
                            h, w = frame_yolo.shape[:2]

                            banner_h = 80

                            # Banner semitransparente en la parte superior
                            #overlay = frame_yolo.copy()
                            #cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 255), -1)
                            #alpha = 0.2
                            #cv2.addWeighted(overlay, alpha, frame_yolo, 1 - alpha, 0, frame_yolo)

                            # Texto "DEFECTO" centrado dentro de la bounding box
                            x1, y1, x2, y2 = results[0].boxes.xyxy[i].tolist()
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            texto = "DEFECTO"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.2
                            thickness = 3
                            (tw, th), baseline = cv2.getTextSize(texto, font, font_scale, thickness)
                            tx = cx - tw // 2
                            ty = cy + th // 2
                            cv2.putText(frame_yolo, texto, (tx, ty), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
                        if class_name == "B":
                            #print(f"🎯 Objeto detectado: {class_name}")
                            count_bueno+=1
                            
                    if bandera_cv == False:
                        cv2.namedWindow("Vista Camara", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow("Vista Camara", WIDTH, HEIGHT)
                        bandera_cv = True

                    frame_yolo_resized = cv2.resize(frame_yolo, (WIDTH, HEIGHT))
                    

                    cv2.imshow("Vista Camara", frame_yolo_resized)
                    #cv2.imshow("Vista Camara",frame_yolo)
                    
                    actualizar_linea_archivo(linea_cero,count_bueno)
                    actualizar_linea_archivo(linea_uno,count_malo)
                    print(f"ELECTRODO BUENO :{count_bueno}, MALO {count_malo}")

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

                #if ultimo_frame is not None :
                    #cv2.imshow("Vista Camara",ultimo_frame)

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
