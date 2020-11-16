import os

dataDir = "./FERET_data"
we_want = ['fa', 'fb', 'ql', 'qr', 'rb', 'rc']
we_dont_want = ['pl', 'hl', 'pr', 'hr', 're', 'ra', 'rd']

allFiles = []
keepFiles = []
deleteFiles = []

imageFolderPath = [os.path.join(dataDir, f) for f in os.listdir(dataDir)]

for imageFolder in imageFolderPath:
    for image in os.listdir(imageFolder):
        if any(angle in image for angle in we_dont_want):
            imagePath = os.path.join(imageFolder, image)
            os.remove(imagePath)
            deleteFiles.append(image)
        allFiles.append(image)

print('//----------------------//')
print("all files counted :" + str(len(allFiles)))
print("files to delete :" + str(len(deleteFiles)))
print("remainingFiles after deleting :" + str(len(allFiles) - len(deleteFiles)))
