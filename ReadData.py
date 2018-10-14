def load_imgs_and_keypoints(dirname='C:\\data\\'):
    # Write your code for loading images and points here
    imgs = []
    #for filename in os.listdir(dirname):
    #    print(dirname,filename)
    #    img = mpimg.imread(os.path.join(dirname,filename))
    #    if img is not None:
    #        images.append(img)
            #plt.imshow(images)
            #plt.show()

    
    #gt=pd.read_csv('C:\\gt.csv', sep=',')

    data = []
    with open("C:\\gt.csv") as f:
        for line in f:
            data.append([x for x in line.split()])
    
    file = open("\\gt.csv", "r")
    data = [map(float, line.split(",")) for line in file]
    
    import csv
 
    with open("C:\\gt.csv") as f:
        data = [map(float, row) for row in csv.reader(f, delimiter=',')]
 
    f = open(u'C:\\gt.csv')
    data = pd.read_csv(f, sep=',')
    
    return imgs,data

imgs, points = load_imgs_and_keypoints()
