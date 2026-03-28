class Config:
    input_size = (224, 224)       
    num_classes = 1000             # ILSVRC 2014 classes

    mapping_resolution = { #outputs resolution
        "alexnet": 13,           
        "vgg": 14,                
        "googlenet": 14           
    }

    cam_threshold = 0.2           
