from flask import Flask,render_template,jsonify,request
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
import pickle as p
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

sub_science = [{'name':'Digital Electronics and Computers'},{'name': 'Mathematics (voc)'},
       {'name':'Physics'}, {'name':'Chemistry'}, {'name':'Computer organization and operating system'},
       {'name':'Electronic Material and Workshop Practice'}, {'name':'Biology'},
       {'name':'Computer science'},{'name': 'Computer software application'},
       {'name':'Electronic and Electrical measuremets'}, {'name':'Marathi'},
       {'name':'Domestic and Consumer Appliances'}, {'name':'Mathematics'},
       {'name':'Mathematics and Statistics'}, {'name':'Sociology'}, {'name':'Web technology'}, {'name':'Hindi'},
       {'name':'Psychology'}]

sub_arts = ['Accommodation Operation', 'Accountancy', 'Bakery',
       'Banking', 'Biology', 'Botany', 'Business Studies', 'Chemistry',
       'Civics', 'Computer Application', 'Co-operation', 'Economics',
       'English', 'French', 'General Foundation Course', 'Hindi', 'History',
       'Konkani', 'Logic', 'Marathi', 'Mathematics', 'Physics',
       'Political Science', 'Portuguese', 'Psychology', 'Secretarial Practice',
       'Sociology']

all_subjects = ['chemistry', 'evs', 'ge:botany', 'maths', 'physics', 'ge:history',
       'ge:electronics', 'ge:psychology', 'ge:english', 'ge:cs',
       'ge:marathi', 'ge:konkani', 'ge:hindi', 'botany', 'ge:maths',
       'ge:philosophy', 'ge:geography', 'cs', 'ge:sociology',
       'ge:marketing management', 'ge:french', 'ge:chemistry', 'ge:ppa',
       'electronics', 'ge:portuguese', 'micro', 'ge:banking',
       'biotechnology', 'botany-biotechnology', 'ge:biotechnology']

all_subjects_arts = ['KOG103', 'KGC101', 'French (GE)', 'KONKANI (SEC)',
       'KOC103  kONKANI kAVITA', 'GE : Banking', 'KOC101',
       'BOG101 Environmental   Biotechnology',
       'GE : Marketing Management', 'Marketing Management - GE',
       'KOC105 Konkani Bhas  ani sahityacho Itihas',
       'KOD101 Prashasakiy Vevharantli Konkani',
       'KOD103 Bakibab Borkar  hanchea  Konkani kavitancho abhyas  ',
       'KOC107 Venchik konkani kadambareancho abhyas',
       'kOC106 BHARTIYA  KAVYASHASTR', 'KOD103  BHASVIDHYANACHI  VALLAKH',
       'KOA101', 'KOG101', 'Introduction to Journalism',
       'Introduction  to  Advertising', 'SOC101',
       'POG101 CONTEMPORARY ISSUES IN INDIA',
       'POC101 INTRODUCTION TO  POLITICAL THEORY', 'SOG101', 'SOC103',
       'POC103 INDIAN  CONSTITUTION',
       'POS104  LEADERSHIP SKILLS IN  POLITICS', 'SOG103',
       'POG107 INTRODUCTION TO  HUMAN RIGHTS', 'SOS103',
       'POD103  Public Administration',
       'POC105  WESTERN POLITICAL THINKERS (PLATO TO  LOCKE) ',
       'POD101  INTERNATIONAL  RELATIONS', 'POD105 COMPARATIVE  GOVT.',
       'POC109  GOVERNMENT  AND  POLITICS  OF  GOA  (1961 - 1987)',
       'POC107 INDIAN  POLITICAL  THINKERS (KAUTILYA TO  VIVEKANANDA)',
       'Reporting  and  Feature  Writing', 'Event Management',
       'Advertising Management', 'TV Anchoring', 'Television Journalism',
       'Media  and Public  Opinion', 'Visual Design',
       'Business  and  Sports  Journalism', 'Photo Journalism',
       'Press Law  & Ethics', 'PROJECT', 'SOC105',
       'SOD101  Indian Society  Issues and Concerns', 'SOC106',
       'SOD102  Rural Society in India', 'SOD103', 'SOC107', 'MRA101 ',
       'MRG101 ', 'MRC101', 'Compulsory  English', 'MRG103', 'MRC103',
       'MRS101 ', 'English  Compulsory', 'MRD103', 'Project',
       'Graphic  Designing', 'Video Editing', 'History of  Media',
       'Film Appreciation', 'TV Production', 'Media Law  and  Ethics',
       'ECC105 INDIAN ECONOMY ', 'ECD107 PUBLIC  FINANCE I',
       'HSD103  Rise  of  the  Modern West',
       'HSC 105  Indian National Movement',
       'ECD109 INTERNATIONAL  ECONOMICS', 'ECC103 MACROECONOMICS',
       'HSC103 History of Medieval India', 'HGC101', 'HNS101', 'HNC103',
       'HND102  ', 'HND101', 'HNC106', 'HND103', 'HNC107', 'HNC105',
       'HSG103 History  of  Human  Civilisation',
       'PID101 Applied  Ethics I', 'PID101  HISTORY OF IDEAS - I',
       'PIC103  WORLD  RELIGIONS', 'HSC101 History  of Goa',
       'PIC101  MORAL PHILOSOPHY - I',
       'HSG102 Indian Culture and Heritage', 'HNA101', 'HNC101',
       'Entrepreneurship  Development (GE)',
       'PIG101 ENVIRONMENTAL ETHICS - I',
       "HSS106 Appreciating India's Heritage",
       'Introduction to  Communication Principles',
       'AECC - English  English  Communication',
       'AECC - Hindi  Sampreshan Kaushal  ',
       'Introduction  to Multimedia', 'ECC101 MICROECONOMICS',
       'Philosophy (GE)', 'Computer Science (SYBA) GE', 'HNG103',
       'Introduction to Audio Video Media', 'Audio Production', 'HNG101',
       'Computer  Science (FYBA)   GE', 'PIS101 PRACTICAL REASONING',
       'GE : Media  and  Psychology', 'ECONOMICS (SEC)',
       'PSC 101 Fundamentals  of  Psychology- I',
       'ENC101 Popular Literature', 'AECC - English',
       'Geography (BA) - GE', 'PRC101 : Civilization and Culture -  I',
       'PSG 101 Child Psychology ', 'PSS 101 Stress Management ',
       'PSC 103 Social Psychology-I', 'ENGLISH  (GE) Womens Writing',
       'PSG 103 Psychology  of  Gender  and Identity',
       'ENC103 British Poetry  and  Drama',
       'GC : Communicative  English 1.1', 'ENGLISH (GE)',
       'MATHEMATICS (GE)', 'English (GENERAL CORE)',
       'ENS102 Creative Writing',
       'PRG103 : Portuguese Language Level III', 'Geography (GE)',
       'MGC101', 'PRC103 : Portuguese : Literary Prose',
       'Portuguese  SEC', 'ENC105 American Literature',
       'END103 Modern  Indian  Writing  in English Translation ',
       'ENC107  British Romantic  Literature',
       'END101  Literary  Criticism',
       'END106  Science And Detective Fiction',
       'ENC106 Modern European  Drama',
       'PSD 104 (Honours)  Criminal  Psychology',
       'PSD 102  Health Psychology',
       'PSD 101 (Honours)  Statistics  for  Psychology',
       'PSC 105 Psychological Disorders',
       'PSC 106 (Honours) Psychological  Testing',
       'PSC 107 (Honours)  Positive Psychology', 'MRD102', 'MRC106',
       'MRC105', 'MRC107', 'MRD101']

label_value = ['Excellent','Good','Average','Bad']
percentage = {"O":"90%","A+":"90%","A":"80%","B+":"70%","AB":"0%","B":"60%","C":"50%","p":"40%","P":"40%","F":"0%","ND":"50%-80%"}
course = ['0','0','0','0','0','0']
course_name = ['PCM','PCB','MCB','PCSE','PCSM']
att_science = [{}]
att_arts = [{}]
data_arts =[{}]
model = []
model1 = '0'
ack = []
log = ''



@app.route('/form', methods=['GET','POST'])
def makecalc():
    if request.method == 'POST':
        global att_science
        data = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        req = request.get_json()
        req = req['user']
        print(req)
        for i in range(0,6):
            for j in range(0,18):
                dictn = sub_science[j]
                
                if(dictn['name'] == req['name'+str(i+1)]):
                    data[0][j] = int(req['value'+str(i+1)])
        count = 0
        print(course_name)
        for i in course_name:
              modelfile = './templates/'+i+'.pickle'
              model.append(p.load(open(modelfile, 'rb')))
        for m in model:
            pr = m.predict(data)
            course[count] = percentage[pr[0]]
            #print(course)
            count +=1
        '''att_science = [{'id':1,'name':'PCM','r':course[0]},{'id':2,'name':'PCB','r':course[1]},
                       {'id':3,'name':'MCB','r':course[2]},{'id':4,'name':'PCSE','r':course[3]},
                       {'id':5,'name':'PCSM','r':course[4]}]'''
        #print(data3)
        data = [[course[0],course[1],course[2],course[3],course[4]]]
        df = pd.DataFrame(data, columns = ['r1','r2','r3','r4','r5'])
        df.to_csv("science_form.csv")
        print(att_science)
        return ''
    else:
        courses = pd.read_csv('./science_form.csv')
        att_science = [{'id':1,'name':'PCM','r':courses['r1'][0]},{'id':2,'name':'PCB','r':courses['r2'][0]},
                       {'id':3,'name':'MCB','r':courses['r3'][0]},{'id':4,'name':'PCSE','r':courses['r4'][0]},
                       {'id':5,'name':'PCSM','r':courses['r5'][0]}]
        print(att_science)
        return jsonify(att_science)

@app.route('/form_arts', methods=['GET','POST'])
def form_arts():
    if request.method == 'POST':
        global data_arts
        global att_arts
        data = []
        data1 = []
        sub_name = ['','','','','',''] 
        label_name = [ 'Hindi_y', 'History_y', 'Political Science_y',
                       'Portuguese_y',  'Economics_y','English_y', 'Marathi_y', 'Konkani_y',
                       'Psychology_y', 'Philosophy', 'Sociology_y']
        req = request.get_json()
        req = req['user']
        print(req)
        for i in range(0,6):
            for j in range(0,27):
                dictn = sub_arts[j]
                
                if(dictn == req['name'+str(i+1)]):
                    data.append(int(req['value'+str(i+1)]))
                    sub_name[i]=dictn
                    break
        course=['0','0','0','0','0','0','0','0','0','0','0']
        table = pd.read_csv('./templates/final.csv')
        data1.append(data)
        for i in range(len(label_name)):
            grades = table[table[label_name[i]]!='ND']
            x_train = grades[[sub_name[0],sub_name[1],sub_name[2],sub_name[3],sub_name[4],sub_name[5]]]
            x_train = x_train.values
            y_train = grades[[label_name[i]]]
            y_train = y_train.values
            cart =  DecisionTreeClassifier()
            cart.fit(x_train,y_train.ravel())
            p = cart.predict(data1)
            print(p[0])
            course[i] = percentage[p[0]]
        '''att_arts = [{'id':1,'name':'Hindi','r':course[0]},{'id':2,'name':'History','r':course[1]},
                {'id':3,'name':'Political science','r':course[2]},{'id':4,'name':'Portuguese','r':course[3]},
                {'id':5,'name':'Economics','r':course[4]},{'id':6,'name':'English','r':course[5]},
                 {'id':7,'name':'Marathi','r':course[6]},
                 {'id':8,'name':'Konkani','r':course[7]},{'id':9,'name':'Psychology','r':course[8]},
                 {'id':10,'name':'Philosophy','r':course[9]},{'id':11,'name':'Sociology','r':course[10]}]'''
        
        data = [[course[0],course[1],course[2],course[3],course[4],course[5],course[6],course[7],course[8],course[9],course[10]]]
        df = pd.DataFrame(data, columns = ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11'])
        df.to_csv("arts_form.csv")
        #print(data3)
        #print(send_data)
        return ''
    else:
        courses = pd.read_csv('./arts_form.csv')
        att_arts = [{'id':1,'name':'Hindi','r':courses['r1'][0]},{'id':2,'name':'History','r':courses['r2'][0]},
                {'id':3,'name':'Political science','r':courses['r3'][0]},{'id':4,'name':'Portuguese','r':courses['r4'][0]},
                {'id':5,'name':'Economics','r':courses['r5'][0]},{'id':6,'name':'English','r':courses['r6'][0]},
                {'id':7,'name':'Marathi','r':courses['r7'][0]},
                {'id':8,'name':'Konkani','r':courses['r8'][0]},{'id':9,'name':'Psychology','r':courses['r9'][0]},
                {'id':10,'name':'Philosophy','r':courses['r10'][0]},{'id':11,'name':'Sociology','r':courses['r11'][0]}]
        return jsonify(att_arts)
    
@app.route('/black', methods=['GET','POST'])
def  att():
    if request.method == 'GET':
        #req = request.get_json()
       # req = req['roll']
        att = pd.read_csv('./templates/final_att.csv')
        st = att[att['studrollno']==log][['june','july','august']]                               
        val = st.values
        if list(val)!= []:
            modelfile = './templates/august.pickle'
            model1=p.load(open(modelfile, 'rb'))
            global ack
            ack = []
            p1 = ''
            print(val)
            p1 = model1.predict(val)
            df2 = pd.read_csv('./templates/attendance.csv')
            df2 = df2[(df2['studrollno']==log)&(df2['mcode']==8)][['SubName', 'no_of_lects_present', 'lects_taken','no_of_lects_absent','fnm']].sort_values('SubName')
            v = df2.values
            l={}
            for i in range(len(v)):
                s = {'id':i,'subname':v[i][0],'lp':v[i][1],'lt':v[i][2],'la':v[i][3],'f':v[i][4]}
                ack.append(s)
            ack.append({'id':'a','ack':p1[0]})
            print(ack)
            return jsonify(ack)
        else:
            black_arts()
            return jsonify(ack)

@app.route('/black_arts', methods=['GET','POST'])
def  black_arts():
    if request.method == 'GET':
        #req = request.get_json()
       # req = req['roll']
        att = pd.read_csv('./templates/final_att_arts.csv')
        st = att[att['studrollno']==log][["june","july","august"]]                               
        val = st.values
        modelfile = './templates/aug_arts_black.pickle'
        model1=p.load(open(modelfile, 'rb'))
        global ack
        ack = []
        p1 = ''
        print(val)
        p1 = model1.predict(val)
        df2 = pd.read_csv('./templates/attendance_arts.csv')
        df2 = df2[(df2['studrollno']==log)&(df2['mcode']==8)][['SubName', 'no_of_lects_present', 'lects_taken','no_of_lects_absent','fnm']].sort_values('SubName')
        v = df2.values
        l={}
        for i in range(len(v)):
            s = {'id':i,'subname':v[i][0],'lp':v[i][1],'lt':v[i][2],'la':v[i][3],'f':v[i][4]}
            ack.append(s)
        ack.append({'id':'a','ack':p1[0]})
        print(ack)
        return jsonify(ack)
    
@app.route('/login', methods=['POST'])
def stud():
    req = request.get_json()
    req = req['roll']
    global log
    log = req['name']
    pas = req['pass']
    att = pd.read_csv('./templates/login.csv')
    st = att[att['studrollno']==log][['password']]
    print("st",st)
    print("list(st)",list(st.values))
    if list(st.values) == []:
        return jsonify([{"id":1,"access":"false"}])
    if pas != st.values[0]:
        return jsonify([{"id":1,"access":"true","pass":"false"}])
    
    return jsonify([{"id":1,"access":"true","pass":"true"}])
    
    

@app.route('/pal_black',methods=['GET'])
def pal_black():
    att = pd.read_csv('./templates/final_att.csv')
    st = att[['june','july','august']]
    r = att[['studrollno']] 
    r = r.values                              
    val = st.values
    modelfile = './templates/august.pickle'
    model1=p.load(open(modelfile, 'rb'))
    p1 = ''
    print(val)
    p1 = model1.predict(val)
    print(p1)
    ack = []
    for i in range(len(p1)):
         s = {'id':i,'studrollno':r[i][0],'ack':p1[i]}
         ack.append(s)
    return jsonify(ack)

@app.route('/pal_black_arts',methods=['GET'])
def pal_black_arts():
    att = pd.read_csv('./templates/final_att_arts.csv')
    st = att[['june','july','august']]
    r = att[['studrollno']] 
    r = r.values                              
    val = st.values
    modelfile = './templates/aug_arts_black.pickle'
    model1=p.load(open(modelfile, 'rb'))
    p1 = ''
    print(val)
    p1 = model1.predict(val)
    print(p1)
    ack = []
    for i in range(len(p1)):
         s = {'id':i,'studrollno':r[i][0],'ack':p1[i]}
         ack.append(s)
    return jsonify(ack)

@app.route('/fac_desk', methods=['POST','GET'])
def fac_desk():
    if request.method == 'POST':
        req = request.get_json()
        req = req['roll']
        global log
        log = req['name']
        pas = req['pass']
        att = pd.read_csv('./templates/faculty_name.csv')
        print(log)
        st = att[att['id']==int(log)][['fnm','password']]
        print(st)
        if list(st.values) == []:
            return jsonify([{"id":1,"access":"false"}])
        print("password",st.values[0][1])
        if pas != st.values[0][1]:
            return jsonify([{"id":1,"access":"true","pass":"false"}])
        
        fnm = st.values[0][0]

       
        sc = pd.read_csv('./templates/eval_faculty.csv')
        ar = pd.read_csv('./templates/eval_faculty_arts2.csv')
        
        fac_sc = sc[sc['fnm']==fnm]['fnm'].count()
        fac_ar = ar[ar['fnm']==fnm]['fnm'].count()
        print(fac_sc)
        science = int(fac_sc)
        arts = int(fac_ar)
        
        data = [[fnm, science,arts]]
        df = pd.DataFrame(data, columns = ["name", "science", "arts"])
        df.to_csv("login.csv")
        
        return jsonify([{"id":1,'name':fnm,"access":"true","pass":"true",}])
    else:
        log_fac = pd.read_csv('./login.csv')
        d = [{'name':log_fac["name"][0],"science":int(log_fac["science"][0]),"arts":int(log_fac["arts"][0])}]
        return jsonify(d)
        
        
@app.route('/fac_sub', methods=['GET'])
def fac_sub():
    log_fac = pd.read_csv('./login.csv')
    df = pd.read_csv('./templates/attendance.csv')
    lst = df[df['fnm']==log_fac["name"][0]]['SubName'].unique()
    #print(lst)
    aug = pd.read_csv('./templates/aug_sub_liking.csv')
    send_data = []
    k=0
    
    for i in lst:
        sub = aug[(aug['SubName']==i)&(aug['per_lect']!=0)].drop(columns=['studrollno','SubName'])
        kmean = KMeans(n_clusters = 4)
        kmean.fit( sub[['per_lect']])
        sub['aug_eval'] = kmean.labels_
        temp = sub
        v = []
        for j in range(0,4):
            val = temp['per_lect'].values
            #print(val)
            if len(val)==0:
                v.append(str(0))
            else:
                label = temp[temp['per_lect']==max(val)]['aug_eval'].unique()
                print(max(val))
                #sub['aug_eval'] = temp['aug_eval'].replace({label[0]:label_value[j]})
                temp = temp[temp['aug_eval']!=label[0]]
                print(label[0],temp['aug_eval'].unique())
                v.append(str(sub[sub['aug_eval']==label[0]]['aug_eval'].count()))
            
        
        send_data.append({'id':k,'name':i,'data':v})
        k=k+1
    print(send_data)
    return jsonify(send_data)

@app.route('/fac_sub_arts', methods=['GET'])
def fac_sub_arts():
    log_fac = pd.read_csv('./login.csv')
    df = pd.read_csv('./templates/attendance_arts.csv')
    lst = df[df['fnm']==log_fac['name'][0]]['SubName'].unique()
    #print(lst)
    aug = pd.read_csv('./templates/aug_sub_liking_arts.csv')
    send_data = []
    k=0
    
    for i in lst:
        sub = aug[(aug['SubName']==i)&(aug['per_lect']!=0)].drop(columns=['studrollno','SubName'])
        kmean = KMeans(n_clusters = 4)
        kmean.fit( sub[['per_lect']])
        sub['aug_eval'] = kmean.labels_
        temp = sub
        v = []
        for j in range(0,4):
            val = temp['per_lect'].values
            #print(val)
            if len(val)==0:
                v.append(str(0))
            else:
                label = temp[temp['per_lect']==max(val)]['aug_eval'].unique()
                print(max(val))
                #sub['aug_eval'] = temp['aug_eval'].replace({label[0]:label_value[j]})
                temp = temp[temp['aug_eval']!=label[0]]
                print(label[0],temp['aug_eval'].unique())
                v.append(str(sub[sub['aug_eval']==label[0]]['aug_eval'].count()))
            
        
        send_data.append({'id':k,'name':i,'data':v})
        k=k+1
    print(send_data)
    return jsonify(send_data)


@app.route('/fac_fac', methods=['GET'])
def fac_fac():
    log_fac = pd.read_csv('./login.csv')
    df = pd.read_csv('./templates/attendance.csv')
    lst = df[df['fnm']==log_fac['name'][0]]['SubName'].unique()
    #print(lst)
    aug = pd.read_csv('./templates/aug_fac_liking.csv')
    send_data = []
    k=0
    
    for i in lst:
        sub = aug[(aug['SubName']==i)&(aug['per_lect']!=0)&(aug['fnm']==log_fac["name"][0])].drop(columns=['studrollno','SubName'])
        kmean = KMeans(n_clusters = 4)
        kmean.fit( sub[['per_lect']])
        sub['aug_eval'] = kmean.labels_
        temp = sub
        v = []
        for j in range(0,4):
            val = temp['per_lect'].values
            #print(val)
            if len(val)==0:
                v.append(str(0))
            else:
                label = temp[temp['per_lect']==max(val)]['aug_eval'].unique()
                print(max(val))
                #sub['aug_eval'] = temp['aug_eval'].replace({label[0]:label_value[j]})
                temp = temp[temp['aug_eval']!=label[0]]
                print(label[0],temp['aug_eval'].unique())
                v.append(str(sub[sub['aug_eval']==label[0]]['aug_eval'].count()))
            
        
        send_data.append({'id':k,'name':i,'data':v})
        k=k+1
    print(send_data)
    return jsonify(send_data)

@app.route('/fac_fac_arts', methods=['GET'])
def fac_fac_arts():
    log_fac = pd.read_csv('./login.csv')
    df = pd.read_csv('./templates/attendance_arts.csv')
    lst = df[df['fnm']==log_fac["name"][0]]['SubName'].unique()
    #print(lst)
    aug = pd.read_csv('./templates/aug_fac_liking_arts.csv')
    send_data = []
    k=0
    
    for i in lst:
        sub = aug[(aug['SubName']==i)&(aug['per_lect']!=0)&(aug['fnm']==log_fac["name"][0])].drop(columns=['studrollno','SubName'])
        kmean = KMeans(n_clusters = 4)
        kmean.fit( sub[['per_lect']])
        sub['aug_eval'] = kmean.labels_
        temp = sub
        v = []
        for j in range(0,4):
            val = temp['per_lect'].values
            #print(val)
            if len(val)==0:
                v.append(str(0))
            else:
                label = temp[temp['per_lect']==max(val)]['aug_eval'].unique()
                print(max(val))
                #sub['aug_eval'] = temp['aug_eval'].replace({label[0]:label_value[j]})
                temp = temp[temp['aug_eval']!=label[0]]
                print(label[0],temp['aug_eval'].unique())
                v.append(str(sub[sub['aug_eval']==label[0]]['aug_eval'].count()))
            
        
        send_data.append({'id':k,'name':i,'data':v})
        k=k+1
    print(send_data)
    return jsonify(send_data)

        
@app.route('/pal_sub', methods=['GET'])
def pal_sub():
    aug = pd.read_csv('./templates/aug_sub_liking.csv')
    send_data = []
    
    for i in range(len(all_subjects)):
        sub = aug[(aug['SubName']==all_subjects[i])&(aug['per_lect']!=0)].drop(columns=['SubName'])
        if len(sub)>=4:
            kmean = KMeans(n_clusters = 4)
            kmean.fit( sub[['per_lect']])
            sub['aug_eval'] = kmean.labels_
            temp = sub
            for j in range(0,4):
                val = temp['per_lect'].values
                #print(val)
                if len(val)==0:
                    break
                else:
                    label = temp[temp['per_lect']==max(val)]['aug_eval'].unique()
                    if j==3:
                        send_data.append({'id':i,'name':all_subjects[i],'data':list(temp[temp['aug_eval']==label[0]]['studrollno'].values)})
                        #print(temp[temp['aug_eval']==max(val)]['studrollno'].values)
                    #sub['aug_eval'] = temp['aug_eval'].replace({label[0]:label_value[j]})
                    temp = temp[temp['aug_eval']!=label[0]]
                    
                    
            
        
            
        #print(send_data)
    return jsonify(send_data)

@app.route('/pal_sub_arts', methods=['GET'])
def pal_sub_arts():
    aug = pd.read_csv('./templates/aug_sub_liking_arts.csv')
    send_data = []
    
    for i in range(len(all_subjects)):
        sub = aug[(aug['SubName']==all_subjects_arts[i])&(aug['per_lect']!=0)].drop(columns=['SubName'])
        if len(sub)>=4:
            kmean = KMeans(n_clusters = 4)
            kmean.fit( sub[['per_lect']])
            sub['aug_eval'] = kmean.labels_
            temp = sub
            for j in range(0,4):
                val = temp['per_lect'].values
                #print(val)
                if len(val)==0:
                    break
                else:
                    label = temp[temp['per_lect']==max(val)]['aug_eval'].unique()
                    if j==3:
                        send_data.append({'id':i,'name':all_subjects_arts[i],'data':list(temp[temp['aug_eval']==label[0]]['studrollno'].values)})
                        #print(temp[temp['aug_eval']==max(val)]['studrollno'].values)
                    #sub['aug_eval'] = temp['aug_eval'].replace({label[0]:label_value[j]})
                    temp = temp[temp['aug_eval']!=label[0]]
                    
                    
            
        
            
        #print(send_data)
    return jsonify(send_data)

@app.route('/pal_fac', methods=['GET'])
def pal_fac():
    aug = pd.read_csv('./templates/aug_fac_liking.csv')
    lst = list(aug['fnm'].unique())
    send_data = []
    
    for i in range(len(lst)):
        sub = list(aug[(aug['per_lect']!=0)&(aug['fnm']==lst[i])]['SubName'].unique())
        v = []
        for k in range(len(sub)):
            per = aug[(aug['SubName']==sub[k])&(aug['per_lect']!=0)&(aug['fnm']==lst[i])].drop(columns=['Unnamed: 0'])
            if(len(per)>=4):
                kmean = KMeans(n_clusters = 4)
                kmean.fit( per[['per_lect']])
                per['aug_eval'] = kmean.labels_
                temp = per
                for j in range(0,4):
                    val = temp['per_lect'].values
                    #print(val)
                    if len(val)==0:
                        break
                    else:
                        label = temp[temp['per_lect']==max(val)]['aug_eval'].unique()
                        if j==3:
                            v.append({'id':k,'name':sub[k],'data':list(temp[temp['aug_eval']==label[0]]['studrollno'].values)})
                            
                            #print(temp[temp['aug_eval']==max(val)]['studrollno'].values)
                        #sub['aug_eval'] = temp['aug_eval'].replace({label[0]:label_value[j]})
                        temp = temp[temp['aug_eval']!=label[0]]
                per = per.drop(columns=['aug_eval'])
            else:
                break

        if v!=[]:
            send_data.append({'id':i,'name':lst[i],'data':v})    
                
        
            
            #print(send_data)
    return jsonify(send_data)

@app.route('/pal_fac_arts', methods=['GET'])
def pal_fac_arts():
    aug = pd.read_csv('./templates/aug_fac_liking_arts.csv')
    lst = list(aug['fnm'].unique())
    send_data = []
    
    for i in range(len(lst)):
        sub = list(aug[(aug['per_lect']!=0)&(aug['fnm']==lst[i])]['SubName'].unique())
        v = []
        for k in range(len(sub)):
            per = aug[(aug['SubName']==sub[k])&(aug['per_lect']!=0)&(aug['fnm']==lst[i])].drop(columns=['Unnamed: 0'])
            if(len(per)>=4):
                kmean = KMeans(n_clusters = 4)
                kmean.fit( per[['per_lect']])
                per['aug_eval'] = kmean.labels_
                temp = per
                for j in range(0,4):
                    val = temp['per_lect'].values
                    #print(val)
                    if len(val)==0:
                        break
                    else:
                        label = temp[temp['per_lect']==max(val)]['aug_eval'].unique()
                        if j==3:
                            v.append({'id':k,'name':sub[k],'data':list(temp[temp['aug_eval']==label[0]]['studrollno'].values)})
                            
                            #print(temp[temp['aug_eval']==max(val)]['studrollno'].values)
                        #sub['aug_eval'] = temp['aug_eval'].replace({label[0]:label_value[j]})
                        temp = temp[temp['aug_eval']!=label[0]]
                per = per.drop(columns=['aug_eval'])
            else:
                break

        if v!=[]:
            send_data.append({'id':i,'name':lst[i],'data':v})    
                
        
            
            #print(send_data)
    return jsonify(send_data)

@app.route('/fac_next', methods=['GET'])
def  fac_next():
    log_fac = pd.read_csv('./login.csv')
    att = pd.read_csv('./templates/eval_faculty.csv')
    st = att[att['fnm']==log_fac["name"][0]][['june','july','august']]
    val = st.values
    send_data = [{}]
    modelfile = './templates/lr.pickle'
    model1=p.load(open(modelfile, 'rb'))
    pf = PolynomialFeatures(degree = 3)
    #global ack
    #ack = []
    #p1 = ''
    print(val)
    if list(val)!=[]:
        x_poly = pf.fit_transform(val)
        p1 = model1.predict(x_poly)
        print(p1)
        lst = list(val[0])
        if p1[0][0]>100:
            p1[0][0]=100
        if p1[0][0]<0:
            p1[0][0] = 0
        lst.append(p1[0][0])
        send_data = [{'id':1,'month':lst}]
    return jsonify(send_data)

@app.route('/fac_next_arts', methods=['GET'])
def  fac_next_arts():
    log_fac = pd.read_csv('./login.csv')
    att = pd.read_csv('./templates/eval_faculty_arts2.csv')
    st = att[att['fnm']==log_fac["name"][0]][['june','july','august']]
    val = st[["august"]].values
    send_data = [{}]
    modelfile = './templates/lr_arts.pickle'
    model1=p.load(open(modelfile, 'rb'))
    #global ack
    #ack = []
    #p1 = ''
    print(val)
    if list(val)!=[]:
        p1 = model1.predict(val)
        print(p1)
        lst = list(st[['june','july','august']].values[0])
        if p1[0][0]>100:
            p1[0][0]=100
        if p1[0][0]<0:
            p1[0][0] = 0
        lst.append(p1[0][0])
        send_data = [{'id':1,'month':lst}]
    return jsonify(send_data)
           
      #site-menu-toggle js-menu-toggle   

if __name__ == '__main__':
    app.run()
