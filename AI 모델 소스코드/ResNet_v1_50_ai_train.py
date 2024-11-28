from ops_1 import *



train_path = '/home/mark11/label/cat/Sequestrum/train/' #경로 마지막에 반드시 '/'를 기입해야합니다.
model_name = 'resnet_v1_50'
epoch = 100

if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                epoch=epoch)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)