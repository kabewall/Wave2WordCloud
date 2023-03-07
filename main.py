import nlplot
import pandas as pd
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import sys
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import MeCab

# output.wavからtranscript.txtに単純書き起こし
wave_filename = "output.wav"
wf = wave.open(wave_filename, "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    sys.exit(1)
model = Model(model_name="vosk-model-ja-0.22")
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)
rec.SetPartialWords(True)
with open("transcription.txt", "wb") as f:
    print("Transcripting...")
    for n in tqdm(range(wf.getnframes() // 4000 + 1)):
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result_str = rec.Result()
            result_json = json.loads(result_str)
            text = result_json["text"].replace(" ", "")
            if text == "":
                continue
            f.write(f"{text}\n".encode("utf-8"))

# transcript.txtから形態素分析
print("形態素分析中")
with open("transcription.txt", 'r', encoding='utf-8') as file:
    lines = []
    for i in file:
        lines.append(i.rstrip('\n'))
df_text = pd.DataFrame(lines, columns=['text'], index=None)   # dataFrame読み込み
df_text['text'].replace('', np.nan, inplace=True)               # 空白をNaNに置き換え
df_text.dropna(subset=['text'], inplace=True)  # Nanを削除 inplace=Trueでdf上書き
# スペース区切り分かち書き


def mecab_analysis(text):
    t = MeCab.Tagger('-Ochasen')
    node = t.parseToNode(text)
    words = []
    stop_words = pd.read_csv("stop_words.csv").T.values.tolist()[0]
    while node:
        if node.surface != "":  # ヘッダとフッタを除外
            features_ = node.feature.split(',')
            word_type = features_[0]
            sub_type = features_[1]
            word = features_[6]

            # 品詞を選択
            if word_type in ["名詞"]:
                if sub_type in ['一般', '固有名詞', '組織名', '地名', '人名', 'サ変名詞']:
                    words.append(word)

            # 動詞、形容詞[基礎型]を抽出（名詞のみを抽出したい場合は以下コードを除く）
            elif word_type in ['動詞', '形容詞'] and not (features_[6] in stop_words):
                words.append(features_[6])

        node = node.next

        if node is None:
            break
    return " ".join(words)


df_text['words'] = df_text['text'].apply(mecab_analysis)


# 全データ・#データサイエンティスト・#kaggleをそれぞれインスタンス化
print("プロット作成中")
npt = nlplot.NLPlot(df_text, target_col='words')
# npt_ds = nlplot.NLPlot(df.query('searched_for == "#データサイエンティスト"'), target_col='hashtags')
# npt_kaggle = nlplot.NLPlot(df.query('searched_for == "#kaggle"'), target_col='hashtags')
stopwords = npt.get_stopword(top_n=2, min_freq=0)
fig_wc = npt.wordcloud(
    width=500,
    height=500,
    max_words=100,
    max_font_size=100,
    colormap='tab20_r',
    stopwords=stopwords,
    mask_file=None,
    save=True
)


print("finish!!!")
