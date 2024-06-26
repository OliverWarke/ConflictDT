import torch
import random
import numpy as np
import pandas as pd
from google.colab import drive
from torch import nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,recall_score,precision_score,f1_score
from transformers import BertConfig
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

drive.mount('/content/drive/')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts=[]
        self.labels=[]
        self.raw_texts=[]
        self.raw_sents=[]
        count=0
        round_r=0
        for index,row in df.iloc[1:].iterrows():
            sent_2=[model_sents.encode(row['text'])]
            dists=reward_func(mean_embeds,torch.tensor(sent_2[0]).to(device))
            #Insert Class name and correct class distance, minimum value, and range. Repeat for all classes in dataset
            input_dist='Class Name:'+ str(int(100*round((dists[0]-min0)/(range0), 2))) +''.join(row['text'])
            sents = [tokenizer(input_dist,
                                padding='max_length',
                                max_length=256,
                                truncation=True,
                                return_tensors="pt")]
            round_r+=1
            count+=1
            if round_r>50:
              print(str(count) + 'out of' + str(len(df)))
              round_r=0
            if(len(sents)>=1):
                self.texts.append(sents)
                self.labels.append(row['label'])
                self.raw_texts.append(row['text'])
                self.raw_sents.append(sent_2)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)
    def get_batch_raw_texts(self,idx):
        return self.raw_texts[idx]
    def get_batch_raw_sents(self,idx):
        return self.raw_sents[idx]
    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])
    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_sents = self.get_batch_raw_sents(idx)
        batch_y = self.get_batch_labels(idx)
        batch_embeds=self.get_batch_raw_texts(idx)
        return batch_texts,batch_y,batch_embeds,batch_sents

#Load Dataset Here
df_train=  pd.read_csv('')
df_val=pd.read_csv('')
df_test = pd.read_csv('')
df_train = df_train.rename(columns={'': 'text','':'label'})
df_test = df_test.rename(columns={'': 'text','':'label'})
df_val = df_val.rename(columns={'': 'text','':'label'})

model_sents = SentenceTransformer('all-mpnet-base-v2')
embed = model_sents.encode(list(df_train['text']))
embed=list(embed)
embed=[l.tolist() for l in embed]
df_train['embed']=pd.Series(embed)

embed_v = model_sents.encode(list(df_val['text']))
embed_v=list(embed_v)
embed_v=[l.tolist() for l in embed_v]
df_val['embed']=pd.Series(embed_v)

embed_t = model_sents.encode(list(df_test['text']))
embed_t=list(embed_t)
embed_t=[l.tolist() for l in embed_t]
df_test['embed']=pd.Series(embed_t)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(seed=42)

class BertTextClassifier(BertForSequenceClassification):
    def __init__(self,text_feature_size, dropout=0.1):
        super().__init__(config)
        self.num_labels=6
        combined_dim = 768
        max_seq_len=256
        self.num_bn = nn.BatchNorm1d(config.numeric_features)

        self.dropout = nn.Dropout(dropout)

        num_classes= torch.tensor(6, dtype=torch.int8)
        self.linear_dec=nn.Linear(768,509)
        self.Embedding=nn.Embedding(768,256)
        self.linear = nn.Linear(768, 6)
        self.fc1 = nn.Linear(combined_dim*max_seq_len, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_id, mask,last_reward,last_action,timestep,input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                class_weights=None,
                output_attentions=None,
                output_hidden_states=None,
                numeric_features=None):


        stacked_mask = mask
        stacked_mask = stacked_mask.squeeze(0)

        stacked_inputs=input_id
        stacked_inputs = stacked_inputs.squeeze(0)

        input_ids=stacked_inputs

        outputs=self.bert(input_ids,attention_mask=stacked_mask,head_mask=head_mask,inputs_embeds=inputs_embeds,output_attentions=output_attentions,output_hidden_states=output_hidden_states)

        cls = outputs[1]
        cls= self.dropout(cls)


        linear_output=self.linear(cls)

        out = self.softmax(linear_output)

        return linear_output

number_classes=6
config = BertConfig.from_pretrained('bert-base-uncased',num_labels=number_classes)
config.numeric_features=number_classes
model = BertTextClassifier.from_pretrained('bert-base-uncased',config=config)

def get_mean_embeds(dataset):
  #dividing training dataset into class datasets
  datasets = {}
  by_class = dataset.groupby('label')
  for groups, data in by_class:
      datasets[groups] = data
  #getting mean embeddings for each class
  #first get the embeddings from each datapoint and concatenate them into one tensor
  mean_embeds={}
  class_texts={}
  for i in range(number_classes):
    for j in (datasets[i]['embed']):
      if(i in class_texts.keys()):
          try:
              class_texts[i] = torch.cat([class_texts[i],(torch.FloatTensor(j)).reshape(1,768)], axis=0)
          except Exception as e:
              print(e)
              pass
      else:
          try:
              class_texts[i] = (torch.FloatTensor(j)).reshape(1,768)
          except Exception as e:
            print(e)

  #get mean embedding tensor from each overall class tensor
  for i in class_texts.keys():
    print(i)
    mean_embeds[i]=(torch.mean(class_texts[i].float(),dim=0)).to(device)
  return (mean_embeds)

mean_embeds = get_mean_embeds(df_train)

#Define a reward function for each class in dataset
#This needs refined
def reward_func0(mean_embeds,embed):
  dists=[]
  cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
  for i in mean_embeds.keys():
    dists.append((cos(embed,mean_embeds[i]).item()))
  return dists[0]

def reward_func1(mean_embeds,embed):
  dists=[]
  cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
  for i in mean_embeds.keys():
    dists.append((cos(embed,mean_embeds[i]).item()))
  return dists[1]

def reward_func2(mean_embeds,embed):
  dists=[]
  cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
  for i in mean_embeds.keys():
    dists.append((cos(embed,mean_embeds[i]).item()))
  return dists[2]

df_train['dist_0'] = df_train['embed'].apply(lambda x: reward_func0(mean_embeds,torch.tensor(x).to(device)))
df_train['dist_1'] = df_train['embed'].apply(lambda x: reward_func1(mean_embeds,torch.tensor(x).to(device)))
df_train['dist_2'] = df_train['embed'].apply(lambda x: reward_func2(mean_embeds,torch.tensor(x).to(device)))

min0=round(df_train['dist_0'].min(),4)
min1=round(df_train['dist_1'].min(),4)
min2=round(df_train['dist_2'].min(),4)

max0=round(df_train['dist_0'].max(),4)
max1=round(df_train['dist_1'].max(),4)
max2=round(df_train['dist_2'].max(),4)

range0=abs(max0)+abs(min0)
range1=abs(max1)+abs(min1)
range2=abs(max2)+abs(min2)

def reward_func(mean_embeds,embed):
  dists=[]
  cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
  for i in mean_embeds.keys():
    dists.append((cos(embed,mean_embeds[i]).item()))
  return dists

set_seed(seed=42)
def train(model, train_data, val_data, learning_rate, epochs,train1,val1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(seed=42)

    if use_cuda:
      model = model.cuda()
      criterion = criterion.cuda()
      torch.set_default_device(device)

    train_dataloader = torch.utils.data.DataLoader(train1, batch_size=1, generator=torch.Generator(device=device),shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val1, batch_size=1, generator=torch.Generator(device=device),shuffle=False)
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0
    c=0
    for epoch_num in range(epochs):
        c+=1
        class_texts={}
        tra=0
        print('starting epoch - ' + str(c))
        ran=0
        logits=0
        labels=0
        final_actions=[]
        final_embeds=[]
        count =0
        for step,batch in enumerate(train_dataloader):

            train_label=batch[1]
            embeds=batch[2]
            sents=batch[3]
            final_embeds.append(embeds)
            timestep = 0
            last_action =1

            while timestep<len(batch[0]):
                try:
                  input_id=batch[0][timestep]['input_ids'].to(device)
                  mask=batch[0][timestep]['attention_mask'].to(device)
                  model.zero_grad()
                  output = model(input_id, mask,timestep,last_action,timestep)
                  last_out = output
                  last_out=torch.tensor(last_out).to(device)
                  timestep+=1
                  train_label_t = train_label.to(device)
                  train_label_t=train_label_t.long()
                  last_action = last_out.argmax(dim=1)
                  del last_out
                except Exception as e:
                  print('Error')
                  print(e)
            try:
                if(logits==0):
                    logits=output
                    labels=train_label_t
            except:
                logits = torch.cat((logits,output))
                labels = torch.cat((labels,train_label_t))
            ran+=1
            if ran >20:
                ran=0
                tra+=1
                print(str(count) + 'out of' + str(len(train_dataloader)))
                count+=20
                model.zero_grad()
                loss = criterion(logits,labels)
                del logits
                del labels
                total_loss_train += loss.item()
                ran=0
                loss.backward(retain_graph=True)
                optimizer.step()
                logits=0
                labels=0
        avg_train_loss = total_loss_train / len(train_dataloader)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        total_acc_train = 0
        total_loss_train = 0
    # Store the loss value for plotting the learning curve.
        val_logits=0
        val_labels=0
        final_actions_v=[]
        final_embeds_v=[]
        with torch.no_grad():
            for step,batch in enumerate(val_dataloader):
                val_label=batch[1]
                final_embeds_v.append(batch[2])
                sents_v=batch[3]

                timestep = 0
                last_reward = 0
                last_action =6
                while timestep<len(batch[0]):
                    mask = batch[0][timestep]['attention_mask'].to(device)
                    input_id = batch[0][timestep]["input_ids"].to(device)
                    model.zero_grad()
                    output = model(input_id, mask,timestep,last_action,timestep)
                    last_out = output
                    last_out=torch.tensor(last_out)
                    timestep+=1
                    val_label_t = val_label.to(device)
                    val_label_t=val_label_t.long()
                    mini = last_out.argmax(dim=1)
                    last_action=mini
                final_actions_v.append(last_action)
                del last_out
                try:
                    if(val_logits==0):
                        val_logits=output
                        val_labels=val_label_t
                except:
                    val_logits = torch.cat((val_logits,output))

                    val_labels = torch.cat((val_labels,val_label_t))
                acc = (val_logits.argmax(dim=1)==val_labels).sum().item()

                total_acc_val += acc

                ran+=1
                if ran >20:
                    ran=0
                    print("Forty")
                    model.zero_grad()
                    loss = criterion(val_logits,val_labels)
                    del val_logits
                    del val_labels

                    total_loss_val += loss.item()

                    optimizer.step()
                    final_actions_v=[]
                    final_embeds_v=[]
                    val_logits=0
                    val_labels=0

            print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy: {total_acc_val / len(val_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}")

EPOCHS = 4

LR = 1e-5

print(len(df_train), len(df_val), len(df_test))

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

train1=Dataset(df_train)
val1=Dataset(df_val)
train(model, df_train, df_val, LR, EPOCHS,train1,val1)

def evaluate(model, test_data):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
      print('cuda')
      model = model.cuda()
      torch.set_default_device(device)
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, generator=torch.Generator(device=device))
    model.eval()
    predictions_labels = []
    true_labels = []
    sent_set=[]
    logs=[]
    test_logits=0
    test_labels=0
    mloss = 1
    total_acc_test = 0
    with torch.no_grad():
        for step,batch in enumerate(test_dataloader):
            sents=batch[3]
            timestep = 0
            last_reward = 0
            last_action =0
            test_label=batch[1]
            while timestep<len(batch[0]):
                try:
                  input_id=batch[0][timestep]['input_ids'].to(device)
                  dists=reward_func(mean_embeds,sents[timestep])
                  mask=batch[0][timestep]['attention_mask'].to(device)
                  model.zero_grad()
                  output = model(input_id, mask,timestep,last_action,timestep)
                except Exception as e:
                  print('Error')
                  print(e)
                last_out = output
                last_out=torch.tensor(last_out, device=device)
                timestep+=1
                test_label_t = test_label.to(device)
                test_label_t=test_label_t.long()
                mini = last_out.argmax(dim=1)

                if(mini == test_label_t):
                    last_reward = 1
                else:
                    last_reward = 0
            acc = (output.argmax(dim=1) == test_label_t).sum().item()

            total_acc_test += acc
            logs.append(last_out)
            sent_set.append(sents[-1])
            true_labels += test_label_t.cpu().numpy().flatten().tolist()
            predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return true_labels, predictions_labels,logs

true_labels, pred_labels,log_set = evaluate(model, df_test)

print('Accuracy:  ',accuracy_score(true_labels,pred_labels))
print('F1 Score:  ',f1_score(true_labels,pred_labels,average ='macro'))
print('Recall:    ',recall_score(true_labels,pred_labels,average ='macro'))
print('Precision: ',precision_score(true_labels,pred_labels,average ='macro'))