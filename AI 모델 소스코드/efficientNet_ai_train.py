from ops_3 import *

train_path = '/home/mark11/label/cat/Blepharitis/train/' #경로 마지막에 반드시 '/'를 기입해야합니다.
# val_path = '/home/mark11/dog&cat/cat/Sequestrum/validation/'
model_name = 'efficientnet'
epoch = 100



if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                epoch=epoch)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)

    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()

train_path = '/home/mark11/label/cat/Keratitis/train/' #경로 마지막에 반드시 '/'를 기입해야합니다.
# val_path = '/home/mark11/dog&cat/cat/Sequestrum/validation/'
model_name = 'efficientnet'
epoch = 100



if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                epoch=epoch)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)

    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()

train_path = '/home/mark11/label/cat/Sequestrum/train/' #경로 마지막에 반드시 '/'를 기입해야합니다.
# val_path = '/home/mark11/dog&cat/cat/Sequestrum/validation/'
model_name = 'efficientnet'
epoch = 100



if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                epoch=epoch)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)

    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()

train_path = '/home/mark11/label/dog/Cataract/train/' #경로 마지막에 반드시 '/'를 기입해야합니다.
# val_path = '/home/mark11/dog&cat/cat/Sequestrum/validation/'
model_name = 'efficientnet'
epoch = 100



if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                epoch=epoch)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)

    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()

train_path = '/home/mark11/label/dog/Epiphora/train/' #경로 마지막에 반드시 '/'를 기입해야합니다.
# val_path = '/home/mark11/dog&cat/cat/Sequestrum/validation/'
model_name = 'efficientnet'
epoch = 100



if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                epoch=epoch)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)

    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()

train_path = '/home/mark11/label/dog/Sclerosis/train/' #경로 마지막에 반드시 '/'를 기입해야합니다.
# val_path = '/home/mark11/dog&cat/cat/Sequestrum/validation/'
model_name = 'efficientnet'
epoch = 100



if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                epoch=epoch)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)

    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()