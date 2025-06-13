import numpy as np
from bidict import bidict
from flask import Flask, render_template, request, session, redirect, url_for
# from flask_cors import CORS
from random import choice
from tensorflow import keras


# custom label encoder (to be used within the model).
# bidict makes reverse lookups easier
from bidict import bidict

ENCODER = bidict({
    'a': 0, 
    'e_i' : 1,
    'o_u': 2
})

from bidict import bidict

CHAR_MAP = bidict({
    'ᜀ': 'a',
    'ᜁ': 'e_i',
    'ᜂ': 'o_u',

    'ᜊ᜔': 'b',     'ᜊ': 'ba',   'ᜊᜒ': 'be_bi',   'ᜊᜓ': 'bo_bu',
    'ᜇ᜔': 'd',     'ᜇ': 'da_ra','ᜇᜒ': 'de_di',   'ᜇᜓ': 'do_du',
    'ᜄ᜔': 'g',     'ᜄ': 'ga',   'ᜄᜒ': 'ge_gi',   'ᜄᜓ': 'go_gu',
    'ᜑ᜔': 'h',     'ᜑ': 'ha',   'ᜑᜒ': 'he_hi',   'ᜑᜓ': 'ho_hu',
    'ᜃ᜔': 'k',     'ᜃ': 'ka',   'ᜃᜒ': 'ke_ki',   'ᜃᜓ': 'ko_ku',
    'ᜎ᜔': 'l',     'ᜎ': 'la',   'ᜎᜒ': 'le_li',   'ᜎᜓ': 'lo_lu',
    'ᜋ᜔': 'm',     'ᜋ': 'ma',   'ᜋᜒ': 'me_mi',   'ᜋᜓ': 'mo_mu',
    'ᜈ᜔': 'n',     'ᜈ': 'na',   'ᜈᜒ': 'ne_ni',   'ᜈᜓ': 'no_nu',
    'ᜅ᜔': 'ng',    'ᜅ': 'nga',  'ᜅᜒ': 'nge_ngi',     'ᜅᜓ': 'ngo_ngu',
    'ᜉ᜔': 'p',     'ᜉ': 'pa',   'ᜉᜒ': 'pe_pi',   'ᜉᜓ': 'po_pu',
    'ᜐ᜔': 's',     'ᜐ': 'sa',   'ᜐᜒ': 'se_si',   'ᜐᜓ': 'so_su',
    'ᜆ᜔': 't',     'ᜆ': 'ta',   'ᜆᜒ': 'te_ti',   'ᜆᜓ': 'to_tu',
    'ᜏ᜔': 'w',     'ᜏ': 'wa',   'ᜏᜒ': 'we_wi',   'ᜏᜓ': 'wo_wu',
    'ᜌ᜔': 'y',     'ᜌ': 'ya',   'ᜌᜒ': 'ye_yi',   'ᜌᜓ': 'yo_yu',
    'ᜇ᜔': 'r',     'ᜇ': 'ra',   'ᜇᜒ': 're_ri',   'ᜇᜓ': 'ro_ru'
})


# used for showing progress during the "full quiz".
# 0 -> unanswered, 1 -> correct, 2 -> incorrect
QUIZ_PROG = [
    ('a', 'ᜀ', 0), ('e_i', 'ᜁ', 0), ('o_u', 'ᜂ', 0),

    ('b', 'ᜊ᜔', 0), ('ba', 'ᜊ', 0), ('be_bi', 'ᜊᜒ', 0), ('bo_bu', 'ᜊᜓ', 0),
    ('d', 'ᜇ᜔', 0), ('da_ra', 'ᜇ', 0), ('de_di', 'ᜇᜒ', 0), ('do_du', 'ᜇᜓ', 0),
    ('g', 'ᜄ᜔', 0), ('ga', 'ᜄ', 0), ('ge_gi', 'ᜄᜒ', 0), ('go_gu', 'ᜄᜓ', 0),
    ('h', 'ᜑ᜔', 0), ('ha', 'ᜑ', 0), ('he_hi', 'ᜑᜒ', 0), ('ho_hu', 'ᜑᜓ', 0),
    ('k', 'ᜃ᜔', 0), ('ka', 'ᜃ', 0), ('ke_ki', 'ᜃᜒ', 0), ('ko_ku', 'ᜃᜓ', 0),
    ('l', 'ᜎ᜔', 0), ('la', 'ᜎ', 0), ('le_li', 'ᜎᜒ', 0), ('lo_lu', 'ᜎᜓ', 0),
    ('m', 'ᜋ᜔', 0), ('ma', 'ᜋ', 0), ('me_mi', 'ᜋᜒ', 0), ('mo_mu', 'ᜋᜓ', 0),
    ('n', 'ᜈ᜔', 0), ('na', 'ᜈ', 0), ('ne_ni', 'ᜈᜒ', 0), ('no_nu', 'ᜈᜓ', 0),
    ('ng', 'ᜅ᜔', 0), ('nga', 'ᜅ', 0), ('nge_ngi', 'ᜅᜒ', 0), ('ngo_ngu', 'ᜅᜓ', 0),
    ('p', 'ᜉ᜔', 0), ('pa', 'ᜉ', 0), ('pe_pi', 'ᜉᜒ', 0), ('po_pu', 'ᜉᜓ', 0),
    ('r', 'ᜇ᜔', 0), ('ra', 'ᜇ', 0), ('re_ri', 'ᜇᜒ', 0), ('ro_ru', 'ᜇᜓ', 0),
    ('s', 'ᜐ᜔', 0), ('sa', 'ᜐ', 0), ('se_si', 'ᜐᜒ', 0), ('so_su', 'ᜐᜓ', 0),
    ('t', 'ᜆ᜔', 0), ('ta', 'ᜆ', 0), ('te_ti', 'ᜆᜒ', 0), ('to_tu', 'ᜆᜓ', 0),
    ('w', 'ᜏ᜔', 0), ('wa', 'ᜏ', 0), ('we_wi', 'ᜏᜒ', 0), ('wo_wu', 'ᜏᜓ', 0),
    ('y', 'ᜌ᜔', 0), ('ya', 'ᜌ', 0), ('ye_yi', 'ᜌᜒ', 0), ('yo_yu', 'ᜌᜓ', 0),
]



# app set up
app = Flask(__name__)
# CORS(app)
app.secret_key = 'something_sneaky'


@app.route('/')
def index():
    session.clear()
    return render_template('index.html')


@app.route('/train', methods=['GET'])
def train_get():

    msg = session.get('msg', '')

    #
    #   OPTION 1:
    #       grab a character that has few instances
    #       in the training data compared to other chars
    #
    labels = np.load("data/labels.npy")
    count = {k: 0 for k in ENCODER.keys()}
    for label in labels:
        count[label] += 1
    count = sorted(count.items(), key=lambda x: x[1])
    # q = choice(count[:len(count)//4])[0]
    q = count[0][0]

    #
    #   OPTION 2:
    #       just choose a random one
    #
    # q = choice( list(ENCODER.keys()) )

    return render_template('train.html', q=q, msg=msg)


@app.route('/train', methods=['POST'])
def train_post():

    label = request.form['question']
    labels = np.load("data/labels.npy")
    labels = np.append(labels, label)

    pixels_str = request.form['pixels']
    pixels = pixels_str.split(',')
    img = np.array(pixels).astype(float).reshape(1, 50, 50)

    all_imgs = np.load("data/imgs.npy")
    all_imgs = np.vstack([all_imgs, img])

    np.save("data/labels.npy", labels)
    np.save("data/imgs.npy", all_imgs)

    session['msg'] = f'"{label}" updated in the training data'

    return redirect(url_for('train_get'))


@app.route('/practice', methods=['GET'])
def practice_get():
    return render_template('practice.html', q=choice(list(ENCODER.keys())))


@app.route('/practice', methods=['POST'])
def practice_post():
    try:
        question = request.form['question']
        pixels_str = request.form['pixels']
        pixels = np.array(pixels_str.split(',')).astype(float) / 255.0  # ✅ only divide here

        img = pixels.reshape(1, 50, 50, 1)

        print("IMAGE SHAPE:", img.shape)
        print("PIXEL RANGE:", img.min(), img.max())  # ✅ Should be between 0.0 and 1.0
        print("First few pixels:", pixels[:10])

        model = keras.models.load_model('scripts/baybayin_model_v2.keras')

        pred = np.argmax(model.predict(img), axis=-1)
        pred = ENCODER.inverse[pred[0]]

        print("PREDICTED:", pred, "| EXPECTED:", question)
        print("PIXEL RANGE:", pixels.min(), pixels.max())
        print("PIXEL SUM:", np.sum(pixels))

        correct = 'yes' if pred == question else 'no'

        return render_template('practice.html', q=choice(list(ENCODER.keys())), correct=correct)

    except Exception as e:
        print("ERROR:", e)
        return render_template('error.html')


@app.route('/quiz', methods=['GET'])
def quiz_get():

    if 'progress' not in session:
        session['progress'] = QUIZ_PROG[:]

    progress = session['progress']

    # get a random character that hasn't been tested yet
    possible = [x[0] for x in progress if x[1] != '' and x[2] == 0]
    if possible:
        q = choice(possible)
    else:
        q = 'done'

    # split the progress array into rows
    # so that it's easier to render
    render_prog = []
    for i in range(0, 50, 10):
        render_prog.append(progress[i:i+10])

    return render_template('quiz.html', q=q, progress=render_prog)


@app.route('/quiz', methods=['POST'])
def quiz_post():

    try:
        # get data from the request
        question = request.form['question']
        pixels_str = request.form['pixels']
        pixels = pixels_str.split(',')
        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        model = keras.models.load_model('scripts/baybayin_model_v2.keras')

        # make a prediction
        pred = np.argmax(model.predict(img), axis=-1)
        pred = ENCODER.inverse[pred[0]]

        # indicator for the `progress` array
        # 0 -> unanswered, 1 -> correct, 2 -> incorrect
        result = 1 if pred == question else 2

        # update the progress
        progress = session['progress']

        val = (question, CHAR_MAP.inverse[question], 0)
        upated_val = (question, CHAR_MAP.inverse[question], result)

        progress[progress.index(val)] = upated_val
        session['progress'] = progress

        return redirect(url_for('quiz_get'))

    except Exception as e:
        print(e)
        return render_template('error.html')
    

@app.route('/debug_test')
def debug_test():
    import numpy as np
    from tensorflow import keras

    try:
        model = keras.models.load_model('scripts/baybayin_model_v2.keras')

        # Load one actual image from your test set, for example
        imgs = np.load("data/imgs.npy").astype("float32") / 255.0
        labels = np.load("data/labels.npy") - 1  # Assuming you shifted earlier

        # Use one known sample
        index = 100  # Any index you know is correct
        img = imgs[index].reshape(1, 50, 50, 1)
        label = labels[index]
        pred = np.argmax(model.predict(img), axis=-1)[0]

        # Reverse map using encoder
        true_label = ENCODER.inverse[label + 1]
        pred_label = ENCODER.inverse[pred + 1]

        print("DEBUG TEST")
        print("Image shape:", img.shape)
        print("Predicted:", pred_label)
        print("Actual:", true_label)

        return f"<p>Predicted: <b>{pred_label}</b><br>Actual: <b>{true_label}</b></p>"

    except Exception as e:
        return f"<p>Error: {e}</p>"


if __name__ == '__main__':
    # these are just settings for developing to allow
    # connection to the dev server from different devices,
    # a real webserver would be used for a legit deployment
    # app.run(host='0.0.0.0', port=8082, threaded=True)
    app.run()