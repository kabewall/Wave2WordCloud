import MeCab

text = "何回もいいけど当たり前けどこれってもうちょっとおもろいなぁと思って何回も噛み締めていいことなんじゃないかって結構持ってるなぁ"
t = MeCab.Tagger('-Ochasen')
node = t.parseToNode(text)

while node:
    if node.surface != "":  # ヘッダとフッタを除外
        features_ = node.feature.split(',')
        word_type = features_[0]
        sub_type = features_[1]
        word = features_[6]

        print(word)

        # 品詞を選択
        # if word_type in ["名詞"]:
        #     if sub_type in ['一般']:
        #         word = node.surface
        #         print(node)
        # words.append(word)

        # 動詞、形容詞[基礎型]を抽出（名詞のみを抽出したい場合は以下コードを除く）
        # elif word_type in ['動詞', '形容詞'] and not (features_[6] in stop_words):
        # words.append(features_[6])

    node = node.next

    if node is None:
        break
