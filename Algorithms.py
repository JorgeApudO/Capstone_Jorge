import numpy as np
import cv2
from skimage import color, morphology
from skimage.segmentation import flood
import matplotlib.pyplot as plt

K_3D = 0.9
PENCIL_TOLERANCE = 0.016
SHOW = False

def degree_to_radian(grado):
    return (np.pi/180) * grado

def radian_to_degree(radian):
    return (180/np.pi) * radian

def Mrot(theta):

    C = np.cos(theta)
    S = np.sin(theta)
    R = np.array([[C, -S], [S, C]])
    return R

def flip_y(pose, Image_Y):
    return np.array([pose[0], Image_Y - pose[1]])

def distance(pose1, pose2):
    return np.linalg.norm(pose1 - pose2)

def Jacobian_inv(info: dict):

    q1 = info["q1"]
    q2 = info["q2"]
    L1 = info["L1"]
    L2 = info["L2"]

    a = - (L1 * np.sin(q1) + L2 * np.sin(q1 + q2))
    b = - (L2 * np.sin(q1 + q2))
    c =  L1 * np.cos(q1) + L2 * np.cos(q1 + q2) 
    d = L2 * np.cos(q1 + q2)

    J = np.array([[a, b], [c, d]])
    Jinv = np.linalg.pinv(J)
    return Jinv

def inverse_kinematics(dimension: int, info: dict):

    if dimension == 2:
        return inverse_kinematics_2D(info)
    elif dimension == 3:
        return inverse_kinematics_3D(info)
    else:
        return None
    
def inverse_kinematics_2D(info: dict):

    pose_deseada = np.array([info["pose_deseada"][0], info["pose_deseada"][1]])
    actual_q1 = info["q1"]
    actual_q2 = info["q2"]
    precision = info["precision"]
    max_steps = info["max_steps"]

    #Calcula la cinematica inversa del brazo robotico
    #pose_deseada: Posicion deseada del extremo del brazo
    actual_pose = pose_calc(2, info)
    Error = pose_deseada - actual_pose
    dist = distance(actual_pose, pose_deseada)

    New_q1 = actual_q1
    New_q2 = actual_q2
    count = 0
    pos_info = {"q1": New_q1, "q2": New_q2, "L1": info["L1"], "L2": info["L2"]}
    while dist > precision and count < max_steps:

        J = Jacobian_inv(New_q1, New_q2)
        correction = np.dot(J, Error)
        New_q1 += correction[0]
        New_q2 += correction[1]

        pos_info["q1"] = New_q1
        pos_info["q2"] = New_q2
        
        Error = pose_deseada - pose_calc(2, pos_info)
        dist = distance(pose_calc(2, pos_info), pose_deseada)
        count += 1

    if dist > precision:
        return np.array([None, None])

    return np.array([New_q1, New_q2])

def inverse_kinematics_3D(info: dict):

    pos_2d = inverse_kinematics_2D(info)
    New_q1 = pos_2d[0]
    New_q2 = pos_2d[1]
    pose_deseada = info["pose_deseada"]
    actual_z = info["z"]
    precision = info["precision"]
    max_steps = info["max_steps"]
    
    actual_pose = pose_calc(3, info)
    Error = pose_deseada - actual_pose
    dist = distance(actual_pose, pose_deseada)
    New_z = actual_z
    
    count = 0
    pos_info = {"q1": New_q1, "q2": New_q2, "L1": info["L1"], "L2": info["L2"], "z": New_z}
    while dist > precision and count < max_steps:

        correction = Error[2] * K_3D
        New_z += correction

        pos_info["z"] = New_z

        Error = pose_deseada - pose_calc(3, pos_info)
        dist = distance(pose_calc(3, pos_info), pose_deseada)
        count += 1
    
    if dist > precision:
        return np.array([None, None])
        
    return np.array([New_q1, New_q2, New_z])

def pose_calc(dimension: int, info: dict):

    if dimension == 2:
        return pose_calc_2D(info)
    elif dimension == 3:
        return pose_calc_3D(info)
    else:
        return None
    
def pose_calc_2D(info: dict):

    q1 = info["q1"]
    q2 = info["q2"]
    L1 = info["L1"]
    L2 = info["L2"]
    pencil_diff = info["pencil_diff"] #Diferencia entre la punta del lapiz y el extremo del brazo horizontalmente

    #Calcula una posicion en x e y del extremo efector segun un q1 y q2 dados
    x = L1 * np.cos(q1) + (L2 - pencil_diff) * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + (L2 - pencil_diff) * np.sin(q1 + q2)
    return np.array([x, y])

def pose_calc_3D(info: dict):

    height = info["height"]
    pencil_height = info["pencil_height"] #Diferencia entre la punta del lapiz y el extremo del brazo verticalmente

    #Calcula una posicion del extremo efector segun q1, q2 y altura dados
    pos_xy = pose_calc_2D(info)
    x = pos_xy[0]
    y = pos_xy[1]
    z = height - pencil_height
    return np.array([x, y, z])

def find_wound(image, info: dict):
    #Calcula la posicion del objetivo en la imagen en pixeles
    
    #CAMBIAR POR METODO DE DETECCION DE HERIDA OFICIAL
    #AQUI TENGO ALGO QUE ENCUENTRA LO NOT VERDE PARA HACERLO SIMPLE Y PODER TESTEAR
    MAX_X = info["MAX_X"]
    MAX_Y = info["MAX_Y"]
    x_values = np.arange(0, MAX_X)
    y_values = np.arange(0, MAX_Y)
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
    object_array = np.zeros((MAX_X, MAX_Y, 3))
    for i in range(MAX_X):
        for j in range(MAX_Y):
            object_array[i, j] = np.array([x_values[i], y_values[j], filtered_image_array[j, i]])
    
    average_x = np.average(object_array[:, :, 0], weights=object_array[:, :, 2])
    average_y = np.average(object_array[:, :, 1], weights=object_array[:, :, 2])
    point_center = np.array([average_x, average_y])
    return point_center

def find_pencil(image, info: dict):
    #Calcula la posicion del extremo efector en la imagen en pixeles

    img_hsv = color.rgb2hsv(image)
    img_hsv_copy = np.copy(img_hsv)
    init_pose = info["center"] - np.array([0, info["MAX_Y"]//4])
    init_pose = list(init_pose)

    mask = flood(img_hsv[..., 2], init_pose, tolerance= PENCIL_TOLERANCE)
    img_hsv[mask, 2] = 1

    mask_postprocessed = morphology.binary_opening(mask, np.ones((3, 3)))
    mask_postprocessed = morphology.binary_closing(mask_postprocessed, morphology.disk(20))
    img_hsv_copy[mask_postprocessed, 2] = 1

    mask_post_array = np.array(mask_postprocessed)
    mask_post_x, mask_post_y = np.nonzero(mask_post_array)

    max_y = np.max(mask_post_y)
    max_y_index = np.argmax(mask_post_y)
    max_x = mask_post_x[max_y_index]

    if SHOW:
        plt.imshow(color.hsv2rgb(img_hsv))
        plt.show() #Muestra la imagen con el flood fill sin postprocesar

        plt.imshow(color.hsv2rgb(img_hsv_copy))
        plt.show() #Muestra la imagen con el flood fill postprocesado

        plt.imshow(color.hsv2rgb(img_hsv_copy))
        plt.plot(max_x, max_y, "og", markersize=3) #Coloca un punto verde en el punto encontrado
        plt.show() #Muestra la imagen con el flood fill postprocesado y el punto encontrado

    return np.array([max_x, max_y])



    
