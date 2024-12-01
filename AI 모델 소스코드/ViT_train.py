from torch.utils.data import DataLoader

from ViT_ops import *

train_path = '/data/dd/train/'  # 경로 마지막에 반드시 '/'를 기입해야 합니다.
epochs = 100

data_path = "data"  # 데이터셋 경로
data_loader = ImportData(data_path)
trainset, testset = data_loader.load_data()

# Create DataLoaders
train_loader = DataLoader(trainset, batch_size=16, shuffle=True)
test_loader = DataLoader(testset, batch_size=16, shuffle=False)

if __name__ == '__main__':
    fine_tuning = FineTuning(train_path=train_path,
                                model_name=model_name,
                                epoch=epochs)
    history = fine_tuning.train()
    fine_tuning.save_accuracy(history)
