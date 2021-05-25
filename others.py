#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary

chars = "abcdefghijklmnopqrstuvwxyz"
char_list = [i for i in chars]
n_letters = len(char_list)

n_layers = 1

five_words = ['basic', 'beach', 'below', 'black', 'brown', 'carry', 'cream', 'drink', 'error', 'event', 'exist',
              'first', 'funny', 'guess', 'human', 'image', 'large', 'magic', 'mouse', 'night', 'noise', 'ocean',
              'often', 'order', 'peace', 'phone', 'print', 'quiet', 'reach', 'rough', 'round', 'scene', 'score',
              'sense', 'skill', 'sleep', 'small', 'storm', 'table', 'think', 'touch', 'twice', 'until', 'upset',
              'voice', 'waste', 'watch', 'white', 'woman', 'young']
n_five_words = len(five_words)


def word_to_onehot(string):
    one_hot = np.array([]).reshape(0, n_letters)
    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=n_letters, dtype=int)
        zero[idx] = 1
        one_hot = np.vstack([one_hot, zero])
    return one_hot


def onehot_to_word(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()]


class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(myRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.V = nn.Linear(self.hidden_size, self.output_size)
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.V(out)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layer, 1, self.hidden_size)


class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.V = nn.Linear(self.hidden_size, self.output_size)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, (hidden,hidden))
        out = self.V(out)
        return out, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layer, 1, self.hidden_size)


def main():
    n_hidden = 50  # 26
    n_output = 26
    lr = 0.001
    epochs = 900
    # RNN model
    # model = myRNN(n_letters, n_hidden, n_output, n_layers)
    # LSTM model
    model = myLSTM(n_letters, n_hidden, n_output, n_layers)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)


    print(model)
    for i in range(epochs):
        total_loss = 0
        for j in range(n_five_words):
            string = five_words[j]
            one_hot = torch.from_numpy(word_to_onehot(string)).type_as(torch.FloatTensor())
            optimizer.zero_grad()
            hidden = model.init_hidden()
            input = one_hot[0:-1]
            input = torch.unsqueeze(input, 1)
            target = np.argmax(one_hot[1:], axis=1)

            output, hidden = model(input, hidden)
            loss = loss_func(output.squeeze(1), target)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print('epoch%d' % i)
            print(total_loss)

        scheduler.step()

    torch.save(model.state_dict(), 'trained.pth')
    model.load_state_dict(torch.load('trained.pth'))

    with torch.no_grad():
        total = 0
        positive = 0
        total_text = 0
        positive_text = 0
        for i in range(n_five_words):
            string = five_words[i]
            one_hot = torch.from_numpy(word_to_onehot(string)).type_as(torch.FloatTensor())
            hidden = model.init_hidden()

            input = one_hot[0:-1]
            input = torch.unsqueeze(input, 1)
            target = np.argmax(one_hot[1:], axis=1)
            output, hidden = model(input, hidden)
            output = output.squeeze()

            output_string = string[0]
            for j in range(output.size()[0]):
                output_string += onehot_to_word(F.softmax(output[j].data))
                total_text += 1
                if string[j + 1] == output_string[-1]:
                    positive_text += 1

            total += 1
            if string[-1] == output_string[-1]:
                positive += 1

            print('%d GT:%s OUT:%s' % (i + 1, string, output_string))

        print('final text accuracy %d/%d (%.4f)' % (positive, total, positive / total))
        print('whole text accuracy %d/%d (%.4f)' % (positive_text, total_text, positive_text / total_text))


if __name__ == '__main__':
    main()

