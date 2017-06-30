from sklearn.metrics import confusion_matrix
import codecs

data_path = './data/nlu.test.string.cnn_format'
result_path = './puyang/nlu.gt_preds_best.test'
cm_path = './puyang/nlu.confusion_matrix_best.test'
bc_path = './puyang/nlu.bad_case_best.test'
labels = [
        'nlu.navigation',
        'nlu.poi',
        'nlu.video',
        'nlu.where',
        'other',
        'public.agenda',
        'public.alarm',
        'public.baike',
        'public.beauty',
        'public.calculator',
        'public.calendar',
        'public.call',
        'public.codematch',
        'public.constellation',
        'public.control',
        'public.conversion',
        'public.countdown',
        'public.coupon',
        'public.current_time',
        'public.daijia',
        'public.delivery',
        'public.flight_info',
        'public.gamescore',
        'public.groupcoupon',
        'public.health',
        'public.housing',
        'public.job',
        'public.joke',
        'public.localevent',
        'public.lottery',
        'public.music',
        'public.news',
        'public.picture',
        'public.podcast',
        'public.poem',
        'public.reading',
        'public.recipe',
        'public.reminder',
        'public.roadcondition',
        'public.shopping',
        'public.smarthome',
        'public.sms',
        'public.stock',
        'public.taxi',
        'public.translate',
        'public.violation',
        'public.weather',
        'public.website',
        'public.wenda',
        'public.xianxing',
        'public.yellowpage',
]

f = open(result_path)
lines = f.readlines()
f.close()
gt_list = []
pred_list = []
bad_case_list = []
line_num = len(lines)
for i in range(line_num):
    line = lines[i]
    line = line.strip('\n').split('\t')
    gt_list.append(line[0])
    pred_list.append(line[1])
    if line[0] != line[1]:
        bad_case_list.append(i)
cm = confusion_matrix(gt_list, pred_list, labels=labels)

# Save confusion matrix
f = open(cm_path, 'w')
label_num =len(labels)
s = '\t\t'
for i in range(label_num):
    s = s + 'd%d\t\t' % i
s = s[:-1]+'\n'
f.write(s)
for i in range(label_num):
    s = 'd%d\t' % i
    for j in range(label_num):
        s = s + '%6.0f\t' % float(cm[i, j])
    s = s[:-1] + '\n'
    f.write(s)
f.close()

# Save bad cases
f = open(data_path)
lines = f.readlines()
f.close()
queries = []
line_num = len(lines)
f = codecs.open(bc_path, 'w', encoding='utf8')
for i in bad_case_list:
    line = lines[i]
    str = unicode(line.split('\t')[1].replace(' ', ''), encoding='utf8')
    str = u'%d\t%s\t%s\t%s\n' % (i, gt_list[i], pred_list[i], str)
    f.write(str)
f.close()

