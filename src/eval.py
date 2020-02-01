debug=False
submission=True
batch_size=256
device='cuda:0'
out='.'

datadir = Path('../input/bengaliai-cv19')
featherdir = Path('../input/bengaliaicv19feather')
outdir = Path('.')

from dataset import *
from model import *


# --- Model ---
device = torch.device(device)
n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_total = n_grapheme + n_vowel + n_consonant
print('n_total', n_total)

from torch.utils.data.dataloader import DataLoader
from chainer_chemistry.utils import save_json, load_json


# --- Prediction ---
traindir = '/kaggle/input/bengaliaicv19-trainedmodels/'
data_type = 'test'
test_preds_list = []


def predict_core(test_images, image_size, threshold,
                 arch, n_total, model_name, load_model_path, batch_size=512, device='cuda:0', **kwargs):
    classifier = build_classifier(arch, load_model_path, n_total, model_name, device=device)
    test_dataset = BengaliAIDataset(
        test_images, None,
        transform=Transform(affine=False, crop=True, size=(image_size, image_size),
                            threshold=threshold, train=False))
    print('test_dataset', len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_pred_proba = classifier.predict_proba(test_loader)
    return test_pred_proba


for i in range(4):
    # --- prepare data ---
    indices = [i]
    test_images = prepare_image(
        datadir, featherdir, data_type=data_type, submission=submission, indices=indices)
    n_dataset = len(test_images)
    print(f'n_dataset={n_dataset}')
    # print(f'i={i}, n_dataset={n_dataset}')
    # test_data_size = 200 if debug else int(n_dataset * 0.9)

    model_preds_list = []
    for j in range(4):
        # --- Depends on train configuration ---
        train_args_dict = load_json(os.path.join(traindir, f'args_{j}.json'))
        train_args_dict.update({
            'load_model_path': os.path.join(traindir, f'predictor_{j}.pt'),
            'device': device,
            'batch_size': batch_size,
            'debug': debug,
        })
        print(f'j {j} updated train_args_dict {train_args_dict}')
        test_preds = predict_core(
                test_images=test_images, n_total=n_total,
                **train_args_dict)

        model_preds_list.append(test_preds)

    # --- ensemble ---
    proba0 = torch.mean(torch.stack([test_preds[0] for test_preds in model_preds_list], dim=0), dim=0)
    proba1 = torch.mean(torch.stack([test_preds[1] for test_preds in model_preds_list], dim=0), dim=0)
    proba2 = torch.mean(torch.stack([test_preds[2] for test_preds in model_preds_list], dim=0), dim=0)
    p0 = torch.argmax(proba0, dim=1).cpu().numpy()
    p1 = torch.argmax(proba1, dim=1).cpu().numpy()
    p2 = torch.argmax(proba2, dim=1).cpu().numpy()
    print('p0', p0.shape, 'p1', p1.shape, 'p2', p2.shape)

    test_preds_list.append([p0, p1, p2])
    if debug:
        break
    del test_images
    gc.collect()


p0 = np.concatenate([test_preds[0] for test_preds in test_preds_list], axis=0)
p1 = np.concatenate([test_preds[1] for test_preds in test_preds_list], axis=0)
p2 = np.concatenate([test_preds[2] for test_preds in test_preds_list], axis=0)
print('concat:', 'p0', p0.shape, 'p1', p1.shape, 'p2', p2.shape)

row_id = []
target = []
for i in tqdm(range(len(p0))):
    row_id += [f'Test_{i}_grapheme_root', f'Test_{i}_vowel_diacritic',
               f'Test_{i}_consonant_diacritic']
    target += [p0[i], p1[i], p2[i]]
submission_df = pd.DataFrame({'row_id': row_id, 'target': target})
submission_df.to_csv('submission.csv', index=False)


submission_df

#################################################################
# Check prediction
#################################################################
train = pd.read_csv(datadir/'train.csv')
pred_df = pd.DataFrame({
    'grapheme_root': p0,
    'vowel_diacritic': p1,
    'consonant_diacritic': p2
})

fig, axes = plt.subplots(2, 3, figsize=(22, 6))
plt.title('Label Count')
sns.countplot(x="grapheme_root",data=train, ax=axes[0, 0])
sns.countplot(x="vowel_diacritic",data=train, ax=axes[0, 1])
sns.countplot(x="consonant_diacritic",data=train, ax=axes[0, 2])
sns.countplot(x="grapheme_root",data=pred_df, ax=axes[1, 0])
sns.countplot(x="vowel_diacritic",data=pred_df, ax=axes[1, 1])
sns.countplot(x="consonant_diacritic",data=pred_df, ax=axes[1, 2])
plt.tight_layout()
plt.show()


train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
sns.distplot(train_labels[:, 0], ax=axes[0], color='green', kde=False, label='train grapheme')
sns.distplot(train_labels[:, 1], ax=axes[1], color='green', kde=False, label='train vowel')
sns.distplot(train_labels[:, 2], ax=axes[2], color='green', kde=False, label='train consonant')
plt.tight_layout()
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
sns.distplot(p0, ax=axes[0], color='orange', kde=False, label='test grapheme')
sns.distplot(p1, ax=axes[1], color='orange', kde=False, label='test vowel')
sns.distplot(p2, ax=axes[2], color='orange', kde=False, label='test consonant')
plt.legend()
plt.tight_layout()
