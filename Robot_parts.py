import numpy as np
import cv2
import Algorithms


class Brazo():

    #Clase que define un brazo robotico de dos grados de libertad 2D
    #Asume que el brazo es un eslabon de longitud L1 y otro de longitud L2
    #El brazo está centrado en el origen

    def __init__(self, length_1: float, length_2: float, precision: float, max_steps = 50000):

        self.L1 = length_1 #Longitud del primer eslabon
        self.L2 = length_2 #Longitud del segundo eslabon
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
        info = {"q1": self.q1, "q2": self.q2}
        return pose_calc(2, info)
    
class Ojos():

    def __init__(self , Image_X: int, Image_Y: int):

        self.is_ready = False #Indica si el punto deseado está justo debajo
        self.MAX_X = Image_X #Ancho de la imagen
        self.MAX_Y = Image_Y #Alto de la imagen
        self.x_values = np.arange(0, self.MAX_X) #Posibles posiciones de x en la imagen
        self.y_values = np.arange(0, self.MAX_Y) #Posibles posiciones de y en la imagen

        self.center = np.array([Image_X/2, Image_Y/2]) #Centro de la imagen
        self.sensor = 0 #Altura medida por el sensor
    
    def find_objective(self):
        #Calcula la posicion del objetivo en la imagen

        image = self.imagen
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([20, 0, 0])
        upper_green = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        #mask_inv = cv2.bitwise_not(mask)
        filtered_image = cv2.bitwise_and(image, image, mask=mask)
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_HSV2BGR)
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Filtro", filtered_image)
        cv2.waitKey(0)

        filtered_image_array = np.array(filtered_image)
        filtered_image_array = filtered_image_array / 255
        print(filtered_image_array.shape)
        object_array = np.zeros((self.MAX_X, self.MAX_Y, 3))
        for i in range(self.MAX_X):
            for j in range(self.MAX_Y):
                object_array[i, j] = np.array([self.x_values[i], self.y_values[j], filtered_image_array[j, i]])
        
        average_x = np.average(object_array[:, :, 0], weights=object_array[:, :, 2])
        average_y = np.average(object_array[:, :, 1], weights=object_array[:, :, 2])
        point_center = np.array([average_x, average_y])

        return point_center

    def get_image(self, imagen):

        self.imagen = cv2.imread(imagen)
        cv2.imshow("Imagen", self.imagen)
        cv2.waitKey(0)

class Robot():

    def __init__(self, brazo: Brazo, ojos: Ojos, precision: float, height: float , max_steps = 100000):

        self.brazo = brazo #Instancia de la clase Brazo
        self.ojos = ojos
        self.precision = precision #Precision de la solucion
        self.max_steps = max_steps #Maximo numero de iteraciones para calcular la posicion
        self.pencil_height = height #Altura del lapiz efector
        self.z = self.height / self.screw_thread #Angulo de la altura del robot, q3 = 0 es en brazo z=0
        self.touch = False #Indica si el brazo tocó el punto deseado
        #Permite cerrar el lazo de control visualmente/contacto y no solo por posición estimada
        self.finised = False
         
    
    def pose(self):
        info = {"q1": self.brazo.q1, "q2": self.brazo.q2, }
        return pose_calc(self.brazo.q1, self.brazo.q2, self.q3)
    
    def error_function(self, pose1, pose2):
        return np.linalg.norm(pose1 - pose2)
    
    def inverse_kinematics(self, pose_deseada):
        #Calcula la cinematica inversa del brazo robotico
        #pose_deseada: Posicion deseada del extremo del brazo
        
        new_q1q2 =  self.brazo.inverse_kinematics(pose_deseada[:2])
        if new_q1q2[0] is None:
            return np.array([None, None, None])
        New_q1 = new_q1q2[0]
        New_q2 = new_q1q2[1]
        
        dist = self.error_function(self.pose(), pose_deseada)
        Error = pose_deseada - self.pose()
        New_q3 = self.q3
        count = 0

        while dist > self.precision and count < self.max_steps and (not self.is_ready):

            New_q3 = 0.25*Error[2]/self.screw_thread + New_q3

            dist = self.error_function(self.pose_calc(New_q1, New_q2, New_q3), pose_deseada)
            count += 1
            Error = pose_deseada - self.pose_calc(New_q1, New_q2, New_q3)

        if dist > self.precision:
            return np.array([None, None, None])
        return np.array([New_q1, New_q2, New_q3])
    
    def center_wound(self):

        #Centra la herida en la vista de la camara
        pixel_wound = self.ojos.find_objective()
        pixel_objective = self.ojos.center

        distance = self.ojos.error_function(pixel_wound, pixel_objective)
        move_vector = pixel_wound - pixel_objective
        move_vector = flip_y(move_vector)

        while distance > self.ojos.precision:

            print(f"Distance: {distance}")






            
    

    
if __name__ == "__main__":

    ojo = Ojos(0.1, 174, 386)
    ojo.get_image("corte_2.jpg")
    ojo.find_objective()
    print("Done")
