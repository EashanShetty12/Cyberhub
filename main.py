import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel

# 1. Load PhishTank data
phish_df = pd.read_csv("/content/verified_online.csv")
phish_df = phish_df[['url']].dropna()
phish_df['label'] = 1

# 2. Add some benign URLs
benign_urls = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
    "https://www.wikipedia.org",
    "https://www.twitter.com",
    "https://www.instagram.com",
    "https://www.linkedin.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.netflix.com",
    "https://www.reddit.com",
    "https://www.stackoverflow.com",
    "https://www.github.com",
    "https://www.dropbox.com",
    "https://www.twitch.tv",
    "https://www.salesforce.com",
    "https://www.coursera.org",
    "https://www.udemy.com",
    "https://www.nytimes.com",
    "https://www.bbc.com",
    "https://www.medium.com",
    "https://www.khanacademy.org",
    "https://www.airbnb.com",
    "https://www.booking.com",
    "https://www.paypal.com",
    "https://www.adobe.com",
    "https://www.spotify.com",
    "https://www.pinterest.com",
    "https://www.ebay.com",
    "https://www.cnn.com",
    "https://www.etsy.com",
    "https://www.imdb.com",
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.quora.com",
    "https://www.yahoo.com",
    "https://www.bloomberg.com",
    "https://www.forbes.com",
    "https://www.theguardian.com",
    "https://www.wsj.com",
    "https://www.weather.com",
    "https://www.zillow.com",
    "https://www.indeed.com",
    "https://www.nike.com",
    "https://www.hulu.com",
    "https://www.chase.com",
    "https://www.bankofamerica.com",
    "https://www.citibank.com",
    "https://www.americanexpress.com",
    "https://www.capitalone.com",
    "https://www.verizon.com",
    "https://www.att.com",
    "https://www.t-mobile.com",
    "https://www.samsung.com",
    "https://www.hp.com",
    "https://www.dell.com",
    "https://www.ibm.com",
    "https://www.intel.com",
    "https://www.nvidia.com",
    "https://www.oracle.com",
    "https://www.sap.com",
    "https://www.salesforce.com",
    "https://www.slack.com",
    "https://www.zoom.us",
    "https://www.skype.com",
    "https://www.whatsapp.com",
    "https://www.telegram.org",
    "https://www.snapchat.com",
    "https://www.tiktok.com",
    "https://www.wechat.com",
    "https://www.baidu.com",
    "https://www.alibaba.com",
    "https://www.taobao.com",
    "https://www.jd.com",
    "https://www.sina.com.cn",
    "https://www.weibo.com",
    "https://www.qq.com",
    "https://www.163.com",
    "https://www.sohu.com",
    "https://www.iqiyi.com",
    "https://www.youku.com",
    "https://www.tmall.com",
    "https://www.xiaomi.com",
    "https://www.huawei.com",
    "https://www.lenovo.com",
    "https://www.meituan.com",
    "https://www.douban.com",
    "https://www.zhihu.com",
    "https://www.vk.com",
    "https://www.yandex.ru",
    "https://www.mail.ru",
    "https://www.ok.ru",
    "https://www.rambler.ru",
    "https://www.lenta.ru",
    "https://www.tut.by",
    "https://www.onliner.by",
    "https://www.sberbank.ru",
    "https://www.gazprombank.ru",
    "https://www.vtb.ru"
]

benign_df = pd.DataFrame({'url': benign_urls, 'label': 0})

# 3. Combine and shuffle dataset
df = pd.concat([phish_df.sample(100, random_state=42), benign_df], ignore_index=True).sample(frac=1, random_state=42)

# 4. Load SecBERT
tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")
model = AutoModel.from_pretrained("jackaduma/SecBERT")
model.eval()  # Evaluation mode

# 5. Dataset wrapper
class URLDataset(Dataset):
    def __init__(self, urls, labels):
        self.urls = urls
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        text = self.urls[idx]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            outputs = model(**{k: v.squeeze(0).unsqueeze(0) for k, v in inputs.items()})
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # [768]
        return embedding, self.labels[idx]

# 6. Prepare training data
dataset = URLDataset(df['url'].tolist(), df['label'].tolist())
train_size = int(0.8 * len(dataset))
train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

# 7. Classifier model
class URLClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

# 8. Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = URLClassifier().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=1e-4)

# 9. Training loop
for epoch in range(40):
    clf.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        preds = clf(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# 10. Evaluation
clf.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        preds = (clf(x) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"\n✅ Validation Accuracy: {correct / total:.2%}")

# 11. Function to check if URL is phishing
def check_url(url, model, tokenizer, clf, device):
    inputs = tokenizer(url, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)

    with torch.no_grad():
        prob = clf(embedding).item()

    if prob > 0.5:
        return f"🚨 Phishing detected! (Confidence: {prob:.2f})"
    else:
        return f"✅ Looks authentic. (Confidence: {1 - prob:.2f})"

# 12. Test on new URLs
print("\n🔍 URL Check Examples:")
print(check_url("http://secure-paypal-login.com", model, tokenizer, clf, device))
print(check_url("https://openai.com", model, tokenizer, clf, device))
