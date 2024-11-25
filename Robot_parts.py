import numpy as np
import cv2
from Algorithms import pose_calc, inverse_kinematics, flip_y, find_wound, find_pencil
import serial

class Brazo():

    #Clase que define un brazo robotico de dos grados de libertad 2D
    #Asume que el brazo es un eslabon de longitud L1 y otro de longitud L2
    #El brazo está centrado en el origen

    def __init__(self, length_1: float, length_2: float, precision: float, pencil_diff: float, camera_dist: float, max_steps = 50000):

        self.L1 = length_1 #Longitud del primer eslabon
        self.L2 = length_2 #Longitud del segundo eslabon
        self.pencil_diff = pencil_diff #Distancia entre la punta del lapiz y el extremo del brazo
        self.camera_dist = camera_dist #Distancia entre la camara y el extremo del brazo
        self.precision = precision #Precision de la solucion

        self._q1 = 0 #Angulo de la articulacion 1
        self._q2 = 0 #Angulo de la articulacion 2

        self.max_steps = max_steps #Numero maximo de iteraciones calculo posición

        self.is_ready = False #Indica si el brazo tocó el punto deseado
        #Permite cerrar el lazo de control visualmente y no solo por posición estimada

    #Correccion angulo q1
    @property
    def q1(self):
        return self._q1
    
    @q1.setter
    def q1(self, rad: float):
        self._q1 =  (rad - np.pi) % (2*np.pi) - np.pi

    #Correccion angulo q2
    @property
    def q2(self):
        return self._q2
    
    @q2.setter
    def q2(self, rad: float):
        self._q2 =  (rad - np.pi) % (2*np.pi) - np.pi
    
    def pose(self):
        #Calcula la posicion actual del brazo robotico
        info = {"q1": self.q1, "q2": self.q2, "L1": self.L1, "L2": self.L2, "pencil_diff": self.pencil_diff, "camera_dist": self.camera_dist}
        return pose_calc(2, info)
    
    def get_angles(self, pose_deseada):

        return inverse_kinematics(2, {"pose_deseada": pose_deseada, "L1": self.L1, "L2": self.L2, "precision": self.precision, "max_steps": self.max_steps, "q1": self.q1, "q2": self.q2, "pencil_diff": self.pencil_diff, "camera_dist": self.camera_dist})
        
    
class Ojos():

    def __init__(self, Image_X: int, Image_Y: int, precision: float):

        self.is_ready = False #Indica si el punto deseado está justo debajo
        self.MAX_X = Image_X #Ancho de la imagen
        self.MAX_Y = Image_Y #Alto de la imagen
        self.precision = precision #Precision de la solucion visual
        self.x_values = np.arange(0, self.MAX_X) #Posibles posiciones de x en la imagen
        self.y_values = np.arange(0, self.MAX_Y) #Posibles posiciones de y en la imagen
        self.center = np.array([Image_X/2, Image_Y/2]) #Centro de la imagen
        
    
    def find_wound_pixel(self, imagen):
        #Calcula la posicion del objetivo en la imagen en pixeles
        info = {"MAX_X": self.MAX_X, "MAX_Y": self.MAX_Y, "x_values": self.x_values, "y_values": self.y_values}
        return find_wound(imagen, info)

    def get_image_file(self, imagen, show = False):

        self.imagen = cv2.imread(imagen)
        if show:
            cv2.imshow("Imagen", self.imagen)
            cv2.waitKey(0)

    def find_pencil_pixel(self, imagen):

        info = {"MAX_X": self.MAX_X, "MAX_Y": self.MAX_Y, "x_values": self.x_values, "y_values": self.y_values, "center": self.center}
        return find_pencil(imagen, info)

class Comunicacion():

    def __init__(self, Tx: int, Rx: int, usb_port: str, timeout : float):

        self.port = usb_port #Puerto USB de la camara
        self.baudrate = 9600 #Baudrate esp32
        self.Tx = Tx #Puerto Tx de la Raspberry
        self.Rx = Rx #Puerto Rx de la Raspberry
        self.timeout = timeout #Tiempo de espera para la comunicacion
        self.ser = serial.Serial(self.port, self.baudrate, timeout = self.timeout) #Instancia del puerto serial 

    def send_data_gpio(self, data):

        message = data + "\r\n"
        message = message.encode("utf-8")
        return self.ser.write(message)
    
    def read_data_gpio(self):

        message = self.ser.readline().decode("utf-8")
        #Hay que definir un sistema de mensajes para la comunicacion para los encoders y sensores de distancia
        print(message)

    def read_usb(self):

        #Metodo para leer la camara, tengo que ver como está en la raspberry
        pass

    def send_motor_data(self, data):

        q1 = data[0]
        q2 = data[1]
        z = data[2]
        message = f"{q1},{q2},{z}"
        self.send_data_gpio(message)


class Robot():

    def __init__(self, brazo: Brazo, ojos: Ojos, comunicacion: Comunicacion, precision: float, height: float, screw_thread : float , max_steps = 100000):

        self.brazo = brazo #Instancia de la clase Brazo
        self.ojos = ojos #Instancia de la clase Ojos
        self.comunicacion = comunicacion #Instancia de la clase Comunicacion
        self.precision = precision #Precision de la solucion
        self.max_steps = max_steps #Maximo numero de iteraciones para calcular la posicion
        self.pencil_height = height #Altura del lapiz efector
        self.screw_thread = screw_thread #Distancia que avanza el tornillo por cada vuelta
        self.z = 0 #Altura del brazo
        self.touch = False #Indica si el brazo tocó el punto deseado dentro de una rutina
        #Permite cerrar el lazo de control visualmente/contacto y no solo por posición estimada
        self.finised = False
         
    
    def pose(self):
        info = {"q1": self.brazo.q1, "q2": self.brazo.q2, "L1": self.brazo.L1, "L2": self.brazo.L2, "pencil_diff": self.brazo.pencil_diff, "camera_dist": self.brazo.camera_dist}
        return pose_calc(3, info)
    
    def get_state(self, pose_deseada):
        return inverse_kinematics({"pose_deseada": pose_deseada, "L1": self.brazo.L1, "L2": self.brazo.L2, "precision": self.precision, "max_steps": self.max_steps, "q1": self.brazo.q1, "q2": self.brazo.q2, "pencil_diff": self.brazo.pencil_diff, "camera_dist": self.brazo.camera_dist, "height": self.pencil_height, "pencil_height": self.pencil_height})
    
    def pixel_to_dist_estimation(self, pixel_positions):

        #Estima la distancia real de los pixeles de una imagen
        dist_pixel = np.linalg.norm(pixel_positions[0] - pixel_positions[1])
        center_point = (pixel_positions[1] + pixel_positions[0]) // 2

        angle = 1 #termino mañana


    def center_wound(self):

        #Centra la herida en la vista de la camara
        image = self.comunicacion.read_usb()
        pixel_wound = self.ojos.find_wound_pixel(image)
        pixel_objective = self.ojos.center

        distance = np.linalg.norm(pixel_wound - pixel_objective)
        move_vector = pixel_wound - pixel_objective
        move_vector = flip_y(move_vector)

        while distance > self.ojos.precision:
            
            desired_pose = self.pose() + move_vector * self.pixel_to_dist_estimation([pixel_wound, pixel_objective])
            self.get_state
            pass



    
if __name__ == "__main__":

    print("Hello World")
