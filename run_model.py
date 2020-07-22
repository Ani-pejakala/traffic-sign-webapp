from PIL import  Image
import numpy
import tensorflow as tf
classes = { 1:['Speed limit (20km/h)','The speed limit sign is a regulatory sign. Speed limit signs are designed to communicate a set legal maximum or minimum speed that vehicles must travel. Drivers must not exceed the limit that the sign designates'],
            2:['Speed limit (30km/h)','The speed limit sign is a regulatory sign. Speed limit signs are designed to communicate a set legal maximum or minimum speed that vehicles must travel. Drivers must not exceed the limit that the sign designates'],
            3:['Speed limit (50km/h)','The speed limit sign is a regulatory sign. Speed limit signs are designed to communicate a set legal maximum or minimum speed that vehicles must travel. Drivers must not exceed the limit that the sign designates'],
            4:['Speed limit (60km/h)','The speed limit sign is a regulatory sign. Speed limit signs are designed to communicate a set legal maximum or minimum speed that vehicles must travel. Drivers must not exceed the limit that the sign designates'],
            5:['Speed limit (70km/h)','The speed limit sign is a regulatory sign. Speed limit signs are designed to communicate a set legal maximum or minimum speed that vehicles must travel. Drivers must not exceed the limit that the sign designates'],
            6:['Speed limit (80km/h)','The speed limit sign is a regulatory sign. Speed limit signs are designed to communicate a set legal maximum or minimum speed that vehicles must travel. Drivers must not exceed the limit that the sign designates'],
            7:['End of speed limit (80km/h)','End of previously set upper-speed limit. '],
            8:['Speed limit (100km/h)','The speed limit sign is a regulatory sign. Speed limit signs are designed to communicate a set legal maximum or minimum speed that vehicles must travel. Drivers must not exceed the limit that the sign designates'],
            9:['Speed limit (120km/h)''The speed limit sign is a regulatory sign. Speed limit signs are designed to communicate a set legal maximum or minimum speed that vehicles must travel. Drivers must not exceed the limit that the sign designates'],
            10:['No passing','No Passing (overtaking) for any vehicle type except one line (track) transport (like motorcycles and mopeds)'],
            11:['No passing veh over 3.5 tons','No passing for vehicles with a total weight of over 3.5 t.'],
            12:['Right-of-way at intersection','Indicates priority, only at the upcoming intersection or crossing.'],
            13:['Priority road','Priority Road starts. This sign indicates a priority at all upcoming intersections and crossings till sign End of priority road. Oncoming traffic must yield.'],
            14:['Yield','Yield right-of-way! Drivers must yield to traffic on the crossing priority road. This is only triangular sign facing the corner downwards, so it can also be recognized from the backside.'],
            15:['Stop','Stop and yield right of way! Driver must stop at this sign and yield to traffic on priority road. This sign impliments "yield right of way" more strictly than the previous one.'],
            16:['No vehicles','No entry for any kind of Vehicle. Motorcycles, Mopeds and bicycles can be pushed by hands with engine turned off.'],
            17:['Veh > 3.5 tons prohibited','No entry for motor vehicles with a maximum authorized mass of more than 3,5 t'],
            18:['No entry','All kind of vehicles are prohibited'],
            19:['General caution','General danger / warning sign. A supplementary sign may explains the nature of danger.'],
            20:['Dangerous curve left','A single curve is approaching in left direction.'],
            21:['Dangerous curve right','A single curve is approaching in right direction.'],
            22:['Double curve','Indicates an approaching double curve - first left'],
            23:['Bumpy road','Indicates rough road ahead'],
            24:['Slippery road','Danger of skidding or slipping. Indicates stretches of road that may be slippery when wet or dirty.'],
            25:['Road narrows on the right','Road narrows from the right side. Yield to oncoming traffic through the narrow passage. In case of one-way-street, zipper rule applies.'],
            26:['Road work','Work in process. Be aware of workers.'],
            27:['Traffic signals','Indicates traffic light ahead.'],
            28:['Pedestrians','Pedestrian may cross the road - installation on the right side of road. (This is not sign for pedestrian corsswalk.)'],
            29:['Children crossing','Pay attention to children - installation on the right side of road.'],
            30:['Bicycles crossing','Be aware of cyclists.'],
            31:['Beware of ice/snow','Beware of an icy road ahead. The road can be slippery.'],
            32:['Wild animals crossing','Indicates wild animals may cross the road.'],
            33:['End speed + passing limits','End of all previously set passing and speed restrictions.'],
            34:['Turn right ahead','Indicates that traffic must turn right (after the sign board)'],
            35:['Turn left ahead','Mandatory direction of travel is left, after the sign board.'],
            36:['Ahead only','Mandatory direction of travel. All vehicles must proceed straight ahead. No turns are permitted'],
            37:['Go straight or right','Mandatory directions of travel, straight ahead or right (after the sign board).'],
            38:['Go straight or left','Mandatory directions of travel, straight ahead or left (after the sign board).'],
            39:['Keep right','Prescribed drive direction around the obstacle. Drive from right of the obstacle.'],
            40:['Keep left','Prescribed drive direction around the obstacle. Drive from left of the obstacle.'],
            41:['Roundabout mandatory','Indicates entrance to a traffic circle (roundabout). Traffic in the circle has the right-of-way. Turn signals are not required when entering a roundabout, but must be used when exiting.'],
            42:['End of no passing','End of all passing restrictions'],
            43:['End no passing veh > 3.5 tons','End of no passing zone for vehicles under 3.5t.'] }

def predict(model_name, image_name):
    model = tf.keras.models.load_model('static/models/'+model_name)
    image = Image.open(image_name)
    image = image.resize((32,32))

    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image,dtype=numpy.float32)

    print(image.shape)
    image=image[...,:3]
    pred = model.predict_classes([image])[0]
    print(classes[pred+1])
    return  classes[pred+1]

'''
def forward_pass(model_name, image_name, style):
    model = cv2.dnn.readNetFromTensorflow("LENET.h5")
    tensorflow==2.0.0
    image = cv2.imread(UPLOAD_PATH + image_name)

    if image.shape[1] > MAX_WIDTH:
        image = imutils.resize(image, width=MAX_WIDTH)
    if image.shape[0] > MAX_HEIGHT:
        image = imutils.resize(image, height=MAX_HEIGHT)
    print(image.shape)
    (h, w) = image.shape[:2]

    blob_image = cv2.dnn.blobFromImage(image, 1.0, size=(w, h), mean=(103.939, 116.779, 123.680), swapRB=False,
                                       crop=False)

    model.setInput(blob_image)
    start = time.time()
    output = model.forward()
    end = time.time()
    print(end-start)
    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    # output/=255.0
    output = output.transpose(1, 2, 0)
    # cv2.imshow("out",output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # out=cv2.convertScaleAbs(output, alpha=(255.0))
    filename, file_extension = os.path.splitext(image_name)
    cv2.imwrite(STYLE_PATH + filename +'_'+ style.split('.')[0] + file_extension, output)
    return output'''

