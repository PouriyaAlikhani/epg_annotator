# https://stackoverflow.com/questions/25521120/store-mouse-click-event-coordinates-with-matplotlib
import math
import pandas as pd
import seaborn as sns
from neo import io
import numpy as np
import os
import pickle 
import matlab.engine

k = 8
t_i = 0
coords = []

if os.path.isdir('C:\\Users\palikh\Downloads\Python_Scripts\\'):
    cwd = 'C:\\Users\palikh\Downloads\Python_Scripts\\'
    ##laptop Spectre
elif os.path.isdir('C:\Users\Pouria\Documents\Python Scripts\JD\\'):
    cwd = 'C:\Users\Pouria\Documents\Python Scripts\JD\\'
    ##laptop JW
elif os.path.isdir('/Downloads/Python_Scripts/'):
    cwd = '/home/hoopoo685/Downloads/Python_Scripts/'
    ##@mimi.mcgill.ca
elif os.path.isdir('/home/2015/palikh/COMP396/shared/pouriya_motifs/Data/'):
    cwd = '/home/2015/palikh/COMP396/shared/JoeDent/Data'

# refer to the files in baseline
d0 = {3:'JD29d003.abf',
        8:'JD29d008.abf',
        13:'JD29d013.abf',
        18:'JD29d018.abf',
        23:'JD29d023.abf',
        28:'JD29d028.abf',
        33:'JD29d033.abf',
        38:'JD29d038.abf',
        43:'JD29d043.abf',
        48:'JD29d048.abf',
        2:'JD29003.abf',
        12:'JD29006.abf',
        22:'JD29007.abf',
        32:'JD29016.abf',
        42:'JD29018.abf'}


# refer to the files in pumping
d1 = {0:'JD29d000 - Copy.abf',
        1:'JD29d001 - Copy.abf',
        2:'JD29d002 - Copy.abf',
        3:'JD29d003 - Copy.abf',
        4:'JD29d004 - Copy.abf',
        5:'JD29d005 - Copy.abf',
        6:'JD29d006 - Copy.abf',
        7:'JD29d007 - Copy.abf',
        8:'JD29d008 - Copy.abf',
        9:'JD29d009 - Copy.abf',
        10:'JD29d010 - Copy.abf',
        11:'JD29d011 - Copy.abf',
        12:'JD29d012 - Copy.abf',
        13:'JD29d013 - Copy.abf',
        14:'JD29d014 - Copy.abf',
        15:'JD29d015 - Copy.abf',
        16:'JD29d016 - Copy.abf',
        17:'JD29d017 - Copy.abf',
        18:'JD29d018 - Copy.abf',
        19:'JD29d019 - Copy.abf',
        20:'JD29d020 - Copy.abf',
        21:'JD29d021 - Copy.abf'}

# refer to the files in last
d2 = {7:'JD29d007.abf',
        12:'JD29d012.abf',
        17:'JD29d017.abf',
        22:'JD29d022.abf',
        27:'JD29d027.abf',
        32:'JD29d032.abf',
        37:'JD29d037.abf',
        42:'JD29d042.abf',
        47:'JD29d047.abf',
        52:'JD29d052.abf'}

d3 = {2:'JD29003.abf',
        12:'JD29006.abf',
        22:'JD29007.abf',
        32:'JD29016.abf',
        42:'JD29018.abf'}

dic = d0




    
def make_df(source):
    """
    DESCRIPTION
        this method reads the source abf file located at the cwd and creates and returns it as a pandas DataFrame
    """
    r = io.AxonIO(filename=cwd + source)
    bl = r.read_block(lazy=False, cascade=True)
    # following prints the voltage values
    # print bl.segments[0].analogsignals
    # print bl.segments[0].eventarrays
    a = np.array(bl.segments[0].analogsignals)
    df = pd.DataFrame(data={'time(ms)':[float(i)/10 for i in range(len(a[0]))], 'voltage(mV)':a[0,:,0]*1000}, index = range(len(a[0])), columns=['time(ms)', 'voltage(mV)'])
    return df





def plot2(df, y):
    """
    DESCRIPTION
        this method receives a dataframe, df, and a columns of the dataframe, y. It then plots the y vs. time, where both y and time are columns of the dataframe.
    """
    colors = ['b', 'y', 'g', 'm', 'k', 'w', 'c']
    sns.plt.clf()
    sns.plt.plot(df['time(ms)'], df['voltage(mV)']*(-1), 'ro')
    for i in range(len(df)/10000):
        t_0 = i*10000
        t_1 = (i+1) * 10000
        sns.plt.plot(df['time(ms)'][t_0:t_1-1], df['voltage(mV)'][t_0:t_1-1]*(-1), color=colors[i%5])
    # sns.plt.xticks(np.arange(0, 300, 1))
    # sns.plt.plot(df['time(ms)'], df['voltage(mV)']*(-1))
    sns.plt.title(y)
    fig_manager = sns.plt.get_current_fig_manager()
    px = fig_manager.canvas.width()
    fig_manager.window.move(px, 1)
    fig_manager.window.showMaximized()
    sns.plt.show()





def c(k = k, dic = dic):
    """
    DESCRIPTION
        this method reads and plots the EEG recording with index 'k' from either of the three dictionaries, d0, d1 or d2.
    """
    subPath = findPath(k=k, dic=dic)
    df = make_df(subPath + dic[k])
    plot2(df, 'JD29d' + str(1000+k)[1:4])



def generate_part(k = k, dic = dic):
    df = pd.DataFrame(columns=['t0', 't1', 't2', 't3'])
    subPath = findPath(k = k, dic = dic)
    prompt = '> '
    if (os.path.isfile(cwd + subPath + dic[k].replace('.abf', '') + '.xlsx')):
        print ("%s already exists. Enter \'y\' to overwrite the existing file?" % (dic[k].replace('.abf', '') + '.xlsx'))
        ans = raw_input(prompt)
        if (ans == 'y'):
            df.to_excel(cwd + subPath + dic[k].replace('.abf', '') + '.xlsx', index=False)
    else:
        df.to_excel(cwd + subPath + dic[k].replace('.abf', '') + '.xlsx', index=False)




def generate_full(k = k, dic = dic):
    subPath = findPath(k = k, dic = dic)
    df1 = pd.read_excel(cwd + subPath + dic[k].replace('.abf', '') + '.xlsx', index=False)
    df = pd.DataFrame(columns=['e1', 'e2', 'E1', 'E2', 'R1', 'R2', 'r1', 'r2', 'Pi1', 'Pi2'], index = df1.index)
    df['e1'] = df1['t0'].copy() * 10000
    df['E2'] = df1['t1'].copy() * 10000
    df['R1'] = df1['t2'].copy() * 10000
    df['r2'] = df1['t3'].copy() * 10000
    df['Pi1'] = [[]] * len(df)
    df['Pi2'] = [[]] * len(df)
    prompt = '> '
    if (os.path.isfile(cwd + subPath + dic[k].replace('.abf', '') + ' Full Stamps - incomplete.cPickle')):
        print ("%s already exists. Enter \'y\' to overwrite the existing file?" % (dic[k].replace('.abf', '') + ' Full Annotation' + '.cPickle'))
        ans = raw_input(prompt)
        if (ans == 'y'):
            df.to_pickle(cwd + subPath + dic[k].replace('.abf', '') + ' Full Stamps - incomplete.cPickle')
    else:
        df.to_pickle(cwd + subPath + dic[k].replace('.abf', '') + ' Full Stamps - incomplete.cPickle')




def pump_plot_minor(k=k, frame = -1, dic=dic, pred_plot = False):
    """
    DESCRIPTION
        this method plots the annotated EEG recording.
    """
    subPath = findPath(k=k, dic=dic)
    color_dict = {0:'w', 1:'orange', 2:'g', 3:'b', 4:'r', 5:'y'}
    dfx = pickle.load(open(cwd + subPath + dic[k].replace('.abf', ' Full Annotation.cPickle'), 'rb'))

    if (frame == -1):
        frame = range(0, 90)
    elif (type(frame) == int):
        frame = [frame - 1]
    else:
        frame = list(np.array(frame)-1)
    # if (subPlot < 1):    
    for subPlot in frame: 
        df = dfx.loc[int((subPlot/90.0)*len(dfx)):int(((subPlot+1)/90.0)*len(dfx))-1]
        dim = {'min_y': df['voltage(mV)'].max()*(-1), 'max_y': df['voltage(mV)'].min()*(-1), 'min_x': df['time(ms)'].min(), 'max_x': df['time(ms)'].max()}
        # temp_df = df.copy()
        # lab_array = [[temp_df.index[0], temp_df['class'].loc[0]]]
        # lab_idx = 0
        # for i in df.index:
        #     if (df['class'].loc[i] != lab_array[lab_idx][1]):
        #         lab_array += [df['time(ms)'].loc[i], df['class'].loc[i]]
        #         lab_idx += 1
        # while (len(temp_df) > 0):
        #     lab_array += [[0, 0]]
        #     lab_idx += 1
        #     lab_array[lab_idx][0] = temp_df[temp_df['class'] != lab_array[lab_idx-1][1]].index[0]
        #     lab_array[lab_idx][1] = temp_df['class'].loc[lab_array[lab_idx][0]]
        #     temp_df.drop(temp_df.index[range(lab_array[lab_idx-1][0], lab_array[lab_idx][0])])
        # for lab_idx in range(len(lab_array)):
        #     curr_idx = df['class'].loc[0]
        #     next_idx = df[df['class'] != curr_idx]['class'].loc[0]
        #     df.drop(temp_df.index)
        I = df[df['class'] == 0]
        e = df[df['class'] == 1]
        E = df[df['class'] == 2]
        P = df[df['class'] == 3]
        R = df[df['class'] == 4]
        r = df[df['class'] == 5]
        sns.plt.close('all')
        fig = sns.plt.figure(figsize=(200, 100))
        sns.set_context("paper", rc={"font.size":50})
        sns.plt.xlim(dim['min_x'], dim['max_x'])
        sns.plt.ylim(dim['min_y'], dim['max_y'])
        sns.plt.scatter(I['time(ms)'], I['voltage(mV)']*(-1), label='I Region', color = color_dict[0])
        sns.plt.scatter(e['time(ms)'], e['voltage(mV)']*(-1), label='e Region', color = color_dict[1])
        sns.plt.scatter(E['time(ms)'], E['voltage(mV)']*(-1), label='E Region', color = color_dict[2])
        sns.plt.scatter(P['time(ms)'], P['voltage(mV)']*(-1), label='P Region', color = color_dict[3])
        sns.plt.scatter(R['time(ms)'], R['voltage(mV)']*(-1), label='R Region', color = color_dict[4])
        sns.plt.scatter(r['time(ms)'], r['voltage(mV)']*(-1), label='r Region', color = color_dict[5])
        sns.plt.suptitle('Recording ID: ' + dic[k] + '\n' + 'Frame %d out of 90' % (subPlot+1) , fontsize=150)
        # sns.plt.legend(loc=2,scatterpoints=100, fontsize=150)
        sns.plt.tick_params(labelsize=100)
        sns.plt.savefig(cwd + subPath + str(k) + '\\' + dic[k].replace('.abf', ' Full Annotation %d.png' % (subPlot+1)))
        sns.plt.close(fig)
    del fig




def pump_plot_major(k=k, frame = -1, dic=dic, legend = 2, n_frames = 30):
    """
    DESCRIPTION
        this method plots the annotated EEG recording.
    """
    subPath = findPath(k=k, dic=dic)
    dfx = pickle.load(open(cwd + subPath + dic[k].replace('.abf', ' Annotation.cPickle'), 'rb'))


    if (frame == -1):
        frame = range(0, n_frames)
    elif (type(frame) == int):
        frame = [frame - 1]
    else:
        frame = list(np.array(frame)-1)
    # if (subPlot < 1):
    print ('Figures saved at:')    
    for subPlot in frame: 
        df = dfx.loc[int((subPlot/float(n_frames)*len(dfx))):int(((subPlot+1)/float(n_frames))*len(dfx))-1]
        dim = {'min_y': df['voltage(mV)'].max()*(-1), 'max_y': df['voltage(mV)'].min()*(-1), 'min_x': df['time(ms)'].min(), 'max_x': df['time(ms)'].max()}
        I = df[df['class'] == 0]
        E = df[df['class'] == 1]
        P = df[df['class'] == 2]
        R = df[df['class'] == 3]
        sns.plt.close('all')
        fig = sns.plt.figure(figsize=(200, 100))
        sns.set_context("paper", rc={"font.size":50})
        sns.plt.xlim(dim['min_x'], dim['max_x'])
        sns.plt.ylim(dim['min_y'], dim['max_y'])
        sns.plt.scatter(I['time(ms)'], I['voltage(mV)']*(-1), label='Interpulse', color = 'g')
        sns.plt.scatter(E['time(ms)'], E['voltage(mV)']*(-1), label='Excitation', color = 'b')
        sns.plt.scatter(P['time(ms)'], P['voltage(mV)']*(-1), label='Plateau', color = 'orange')
        sns.plt.scatter(R['time(ms)'], R['voltage(mV)']*(-1), label='Repolarization', color = 'r')
        sns.plt.suptitle('Recording ID: ' + dic[k] + '\n' + 'Frame %d out of %d' % (subPlot+1, n_frames) , fontsize=150)
        if (legend):
            sns.plt.legend(loc=legend,scatterpoints=100, fontsize=150)
        sns.plt.tick_params(labelsize=100)
        print (cwd + subPath + str(k) + '\\' + dic[k].replace('.abf', ' Annotation %d.png' % (subPlot+1)))
        sns.plt.savefig(cwd + subPath + str(k) + '\\' + dic[k].replace('.abf', ' Annotation %d.png' % (subPlot+1)))
        sns.plt.close(fig)
    del fig


def reverse_annotate(k = k, dic = dic):
    """
    DESCRIPTION
        this method takes the dictionary (dic) and trial number (k). It opens stamp2 file with transition timestamps and produces a full annotation of the trial.
    """
    subPath = findPath(k=k, dic=dic)
    stamp = loadR2(k=k, dic=dic)
    df = make_df(subPath + dic[k])
    v = np.full(len(df), 0)
    for i in stamp.index:
        v[int(stamp['e1'].loc[i]*10):int(stamp['E2'].loc[i]*10)] += 1 
        v[int(stamp['E2'].loc[i]*10):int(stamp['R1'].loc[i]*10)] += 2
        v[int(stamp['R1'].loc[i]*10):int(stamp['r2'].loc[i]*10)] += 3

    df['class'] = v
    pickle.dump(df, open(cwd + subPath + dic[k].replace('.abf', ' Annotation.cPickle'), 'wb'))




def annotate(k = k, dic = dic):
    """
    DESCRIPTION
        this method takes the dictionary (dic) and trial number (k). It opens the excel file with transition timestamps and produces a full annotation of the trial.
    """
    df = make_df(subPath + dic[k])
    stamp = pd.read_excel(cwd + subPath + dic[k].replace('.abf', '.xlsx'))

    v = np.full(len(df), 0)
    for i in stamp.index:
        # v[int(stamp['t0'].loc[i]*10):int(stamp['t1'].loc[i]*10)] += 1 
        # v[int(stamp['t1'].loc[i]*10):int(stamp['t2'].loc[i]*10)] += 2
        # v[int(stamp['t2'].loc[i]*10):int(stamp['t3'].loc[i]*10)] += 3
        v[int(stamp['t0'].loc[i]*10000):int(stamp['t1'].loc[i]*10000)] += 1 
        v[int(stamp['t1'].loc[i]*10000):int(stamp['t2'].loc[i]*10000)] += 2
        v[int(stamp['t2'].loc[i]*10000):int(stamp['t3'].loc[i]*10000)] += 3
        if len(v[v > 3]) > 0:
            print i
            return 'boo'
    df['class'] = v
    pickle.dump(df, open(cwd + subPath + dic[k].replace('.abf', ' Annotation.cPickle'), 'wb'))
    print("Major annotation stored at %s" % (cwd + subPath + dic[k].replace('.abf', ' Annotation.cPickle')))
    return df




def annotate_full(k=k, dic=dic):
    subPath = findPath(k=k, dic=dic)
    df = make_df(subPath + dic[k])
    stamp2 = pd.read_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Stamps - incomplete.cPickle'))
    
    v = np.full(len(df), 0)
    for i in stamp2.index:
        if len(v[v[int(stamp2['e1'].loc[i]*10):int(stamp2['r2'].loc[i]*10)] != 0 ]):
            print 'Duplicate Region Index:', i
        v[int(stamp2['e1'].loc[i]*10):int(stamp2['e2'].loc[i]*10)] += 1 
        v[int(stamp2['E1'].loc[i]*10):int(stamp2['E2'].loc[i]*10)] += 2 
        v[int(stamp2['R1'].loc[i]*10):int(stamp2['R2'].loc[i]*10)] += 4 
        v[int(stamp2['r1'].loc[i]*10):int(stamp2['r2'].loc[i]*10)] += 5 
        p_peak = pd.DataFrame(columns = ['start', 'end'], index = range(len(stamp2['Pi1'].loc[i])))
        p_peak['start'] = stamp2['Pi1'].loc[i]
        p_peak['end'] = stamp2['Pi2'].loc[i]
        for j in p_peak.index:
            v[int(p_peak['start'].loc[j]*10):int(p_peak['end'].loc[j]*10)] += 3
    df['class'] = v

    if (os.path.isfile(cwd + subPath + dic[k].replace('.abf', ' Full Annotation.cPickle'))):
        prompt = '> '
        print('\'%s\' already exists, press \'y\' to overwrite' % (dic[k].replace('.abf', ' Full Annotation.cPickle')))
        ans = raw_input(prompt)
        if (ans == 'y'):
            df.to_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Annotation.cPickle'))
    else:
        df.to_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Annotation.cPickle'))


def integ(k, dic = d0):
    """
    DESCRIPTION
        this method returns the integration
    """
    if (len(dic) > 10):
        df = make_df('data\Last\pumping\\' + dic[k])
    elif (k%10 == 2 or k%10 == 7):
        df = make_df('data\Last\\' + dic[k])
    else:
        df = make_df('data\\Baseline\\' + dic[k])
    df['voltage(mV)'] *= 0.0000001
    v = np.array[np.trapz(df['voltage(mV)'][:i] for i in range(len(df['voltage(mV)'])))]
    df['voltage(mV)'] = v
    plot2(df, 'integ')
    return df


def findPath(k = k, dic = dic):
    """
    DESCRIPTION
        this method returns the subpath of the each file
    """
    return 'data\\Baseline\\'
    # if (len(dic) < 6):
    #     return 'data\\Pouriya practice\\'
    # elif (len(dic) > 10):
    #     return 'data\Last\pumping\\'
    # elif (k%10 == 2 or k%10 == 7):
    #     return 'data\Last\\'
    # else:
    #     return 'data\\Baseline\\'


def raw_major(k = k, dic = dic):
    """
    DESCRIPTION
        this method returns the raw data as a dataframe
    """
    subPath = findPath(k, dic)
    if os.path.isfile("%s%s%s" % (cwd, subPath, dic[k].replace('.abf', ' Annotation.cPickle'))):
        df = pickle.load(open(cwd + subPath + dic[k].replace('.abf', ' Annotation.cPickle'), 'rb'))
    else:
        df = make_df(subPath + dic[k])
    stamp = pd.read_excel(cwd + subPath + dic[k].replace('abf', 'xlsx'))
    print 'You have loaded the stamp file and the recording for %s' % dic[k]
    return df, stamp

def raw_minor(temp_k = k, temp_dic = dic):
    """
    DESCRIPTION
        this method returns the raw data (i.e. 1. timestamps and 2. annotations) as a dataframe
    """
    global k, dic
    k = temp_k
    dic = temp_dic
    return loadAnnot(k=k, dic=dic), loadR2(k=k, dic=dic)



def find_stamp_idx(t, t2):
    """
    DESCRIPTION
        this module takes the identity attributes of a pump and returns its acosiated indeces
    """
    t *= 1000
    t2 *= 1000
    idx_set = set()
    idx_set = idx_set.union(stamp2[stamp2['e1'] > t][stamp2[stamp2['e1'] > t]['e1'] < t2].index.tolist())
    idx_set = idx_set.union(stamp2[stamp2['E2'] > t][stamp2[stamp2['E2'] > t]['E2'] < t2].index.tolist())
    idx_set = idx_set.union(stamp2[stamp2['R1'] > t][stamp2[stamp2['R1'] > t]['R1'] < t2].index.tolist())
    idx_set = idx_set.union(stamp2[stamp2['r2'] > t][stamp2[stamp2['r2'] > t]['r2'] < t2].index.tolist())
    return idx_set


def find_major(t=t_i, shift = 0, length = 1.5, k = k, dic = dic):
    """
    DESCRIPTION
        this method finds the 
    """
    global stamp, coords, t_i
    colors = ['b', 'y', 'g', 'm', 'k', 'w', 'c']
    coords = []
    t_0 = int((t+shift)*10000)
    if (length == 1.5):
        t_1 = int((t+shift+0.5)*10000)
    t_2 = int((t+shift+length)*10000)
    fig = sns.plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print('time: %d'%t)
    t_i = t + 5
    t_i = int(t + 1.9)
    sns.plt.clf()
    if (length==1.5):
        sns.plt.plot(df['time(ms)'][t_0:t_1-1], df['voltage(mV)'][t_0:t_1-1]*(-1), color='b')
        sns.plt.plot(df['time(ms)'][t_1:t_2-1], df['voltage(mV)'][t_1:t_2-1]*(-1), color='y')
    else:
        sns.plt.plot(df['time(ms)'][t_0:t_2], df['voltage(mV)'][t_0:t_2]*(-1), 'ro')
        for color_i in range(int(length)):
            t_0 = int((t+shift+color_i)*10000)
            t_2 = int((t+shift+color_i+1)*10000)
            sns.plt.plot(df['time(ms)'][t_0:t_2-1], df['voltage(mV)'][t_0:t_2-1]*(-1), color=colors[color_i%7])
        if (length - int(length) != 0):
            t_0 = int((t+shift+int(length))*10000)
            t_2 = int((t+shift+length)*10000)
            sns.plt.plot(df['time(ms)'][t_0:t_2-1], df['voltage(mV)'][t_0:t_2-1]*(-1), color=colors[int(length)%7])
    sns.plt.title(dic[k] + str(k))
    sns.plt.show()
    fig.canvas.mpl_disconnect(cid)

    if (len(coords) % 4 == 0):
        for i in range(len(coords)/4):
            if (coords[4*i]<coords[4*i+1] and coords[4*i+1]<coords[4*i+2] and coords[4*i+2]<coords[4*i+3]):
                temp = {'t0':coords[4*i], 't1':coords[4*i+1], 't2':coords[4*i+2], 't3':coords[4*i+3]}
                stamp.loc[len(stamp)] = temp
            else:
                print('Invalid Assignment')
                break





def find4(t, shift, length , p1, p2, k = k, dic = dic, scale_y = 0.8):
    global stamp, coords
    coords = []
    max_time = max(df['time(ms)'])
    t_1 = int((t)*10000)
    t_2 = int((t+length)*10000)
    p_1 = int(p1 * 10000)
    p_2 = int(p2 * 10000)
    if (t - shift < 0):
        t_0 = 0
        t_3 = int((t+shift+length)*10000)
    elif (max_time < t+shift+length):
        t_0 = int((t-shift)*10000)
        t_3 = int(max_time*10000)
    else:
        t_0 = int((t-shift)*10000)
        t_3 = int((t+shift+length)*10000)
    fig = sns.plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print('time: %d'%t)
    sns.plt.clf()
    sns.plt.plot(df['time(ms)'][t_0:t_1-1], df['voltage(mV)'][t_0:t_1-1]*(-1), color='y')
    sns.plt.plot(df['time(ms)'][t_1:p_1-1], df['voltage(mV)'][t_1:p_1-1]*(-1), color='r')
    sns.plt.plot(df['time(ms)'][p_1:p_2-1], df['voltage(mV)'][p_1:p_2-1]*(-1), color='g')
    sns.plt.plot(df['time(ms)'][p_2:t_2-1], df['voltage(mV)'][p_2:t_2-1]*(-1), color='r')
    sns.plt.plot(df['time(ms)'][t_2:t_3-1], df['voltage(mV)'][t_2:t_3-1]*(-1), color='b')
    # min_x,max_x = sns.plt.xlim()
    # min_y,max_y = sns.plt.ylim()    
    # length_y = max_y - min_y
    # min_y -= scale_y / 2 * length_y
    # min_y += scale_y / 2 * length_y
    # sns.plt.plot([min_x,max_x],[min_y,max_y])
    # y_min, y_max = df['voltage(mV)'][t_0:t_3-1]*(-1).min(), df['voltage(mV)'][t_0:t_3-1]*(-1).max()
    # sns.plt.
    # sns.plt.title("File: %d\nIDX: %d/%d" % (k))
    sns.plt.title("File: %d" % (k))
    fig_manager = sns.plt.get_current_fig_manager()
    px = fig_manager.canvas.width()
    fig_manager.window.move(px, 1)
    # fig_manager.window.move(px, 5)
    fig_manager.window.showMaximized()
    sns.plt.show()
    fig.canvas.mpl_disconnect(cid)
    return (coords)





def plot_indv_pump(margin = 0.5, k = k, dic = dic, frame = -100):
    subPath = findPath(k=k, dic=dic)
    coords = []
    max_time = max(df['time(ms)'])
    if (frame == -100):
        frame = stamp2.index
    elif (str(type(frame)) == '<type \'int\'>' or str(type(frame)) == '<type \'float\'>'):
        frame = [frame]
    for i in frame:
        starttime = stamp2['e1'].loc[i]/1000
        endtime = stamp2['r2'].loc[i]/1000
        e1 = int(stamp2['e1'].loc[i]*10)
        e2 = int(stamp2['e2'].loc[i]*10)
        E1 = int(stamp2['E1'].loc[i]*10)
        E2 = int(stamp2['E2'].loc[i]*10)
        R1 = int(stamp2['R1'].loc[i]*10)
        R2 = int(stamp2['R2'].loc[i]*10)
        r1 = int(stamp2['r1'].loc[i]*10)
        r2 = int(stamp2['r2'].loc[i]*10)
        
        fig = sns.plt.figure()
        print('time: %d' % starttime)
        sns.plt.clf()

        if (len(stamp2['Pi1'].loc[i]) > 0):
            rth = 0
            P1 = int(stamp2['Pi1'].loc[i][0]*10)
            sns.plt.plot(df['time(ms)'][E2:P1-1], df['voltage(mV)'][E2:P1-1]*(-1), color='k')
            while(rth < len(stamp2['Pi1'].loc[i])):
                P1 = int(stamp2['Pi1'].loc[i][rth]*10)
                P2 = int(stamp2['Pi2'].loc[i][rth]*10)
                sns.plt.plot(df['time(ms)'][P1:P2-1], df['voltage(mV)'][P1:P2-1]*(-1), color='green', label = 'P')
                try:
                    I1 = int(stamp2['Pi2'].loc[i][rth]*10)
                    I2 = int(stamp2['Pi1'].loc[i][rth+1]*10)
                    sns.plt.plot(df['time(ms)'][I1:I2-1], df['voltage(mV)'][I1:I2-1]*(-1), color='k')
                except IndexError:
                    if (rth != len(stamp2['Pi1'].loc[i]) - 1):
                        print ('Baseline Index Error! at index: %d pump: %d' % (rth, i))
                    pass
                rth += 1
            P2 = int(stamp2['Pi2'].loc[i][-1]*10)
            sns.plt.plot(df['time(ms)'][P2:R1-1], df['voltage(mV)'][P2:R1-1]*(-1), color='k')
        else:
            sns.plt.plot(df['time(ms)'][E2:R1-1], df['voltage(mV)'][E2:R1-1]*(-1), color='k')
        if (starttime - margin < 0):
            t_0 = 0
            t_3 = int((endtime+margin)*10000)
        elif (max_time < endtime+margin):
            t_0 = int((starttime-margin)*10000)
            t_3 = int(max_time*10000)
        else:
            t_0 = int((starttime-margin)*10000)
            t_3 = int((endtime+margin)*10000)

        sns.plt.plot(df['time(ms)'][t_0:e1-1], df['voltage(mV)'][t_0:e1-1]*(-1), color='k')
        sns.plt.plot(df['time(ms)'][e1:e2-1], df['voltage(mV)'][e1:e2-1]*(-1), color='mediumvioletred', label = 'e')
        sns.plt.plot(df['time(ms)'][e2:E1-1], df['voltage(mV)'][e2:E1-1]*(-1), color='k')
        sns.plt.plot(df['time(ms)'][E1:E2-1], df['voltage(mV)'][E1:E2-1]*(-1), color='r', label = 'E')
        sns.plt.plot(df['time(ms)'][R1:R2-1], df['voltage(mV)'][R1:R2-1]*(-1), color='b', label = 'R')
        sns.plt.plot(df['time(ms)'][R2:r1-1], df['voltage(mV)'][R2:r1-1]*(-1), color='k')
        sns.plt.plot(df['time(ms)'][r1:r2-1], df['voltage(mV)'][r1:r2-1]*(-1), color='cornflowerblue', label = 'r')
        sns.plt.plot(df['time(ms)'][r2:t_3-1], df['voltage(mV)'][r2:t_3-1]*(-1), color='k', label = 'I')
        sns.plt.title(dic[k] + ' - Pump Index ' + str(i))
        # sns.plt.legend(loc=2))
        sns.plt.savefig(cwd + subPath + str(k) + '\\Individual Plots\\' + dic[k].replace('.abf','') + ' - margin %.1f Pump %d.PNG' % (margin, i))
        sns.plt.clf()
        del fig





def saveR(k = k, dic = dic):
    """
    DESCRIPTION
        this method saves stamp
    """
    global stamp
    if (len(dic) > 10):
        stamp = stamp.sort_values('t0')
        stamp.to_excel(cwd + 'data\Last\pumping\\' + dic[k].replace('abf', 'xlsx'))
    elif (k % 10 == 2 or k % 10 == 7):
        stamp = stamp.sort_values('t0')
        stamp.to_excel(cwd + 'data\Last\\' + dic[k].replace('abf', 'xlsx'))
    else:
        stamp = stamp.sort_values('t0')
        stamp.to_excel(cwd + 'data\\Baseline\\' + dic[k].replace('abf', 'xlsx'))
    annotate(k = k, dic = dic)




def onclick(event):
    global ix
    global coords
    global typekey
    typekey = event.button
    try:
        ix = float(event.xdata)*10
    except TypeError:
        print ("Press \'y\' to Exit")
        prompt = '> '
        ans = raw_input(prompt)
        if ans == 'y':
            return ['quit']

    # except TypeError:
    #     print ("Press \'y\' to Exit")
    #     prompt = '> '
    #     ans = raw_input(prompt)
    #     if ans == 'y':
    #         return ['quit']
    # else:
    #     ix = event.xdata
    print '%d. x = %f'%(len(coords), ix)

    coords.append(ix)

    # if len(coords) == 4 * n:
    #     fig.canvas.mpl_disconnect(cid)

    return coords



def autoEPG(k = k, dic = dic, FileName = -1):
    """
    DESCRIPTION
        this method receives the Peaks(in mV) and their corresponding annotation from AutoEPG software and returns how many we identified in the correct region.
    """
    PathName = cwd + findPath(k=k, dic=dic)
    if (FileName == -1):
        FileName = dic[k]
    eng = matlab.engine.start_matlab()
    eng.cd("C:\\Users\\Pouria\\Google Drive\\#0 Projects\\JoeDent\\", nargout=0)
    # print PathName, FileName
    Peak, Annot = eng.MyEPGAnalysis(PathName, FileName, nargout = 2)
    Peak = Peak[0]
    del eng
    return Peak, Annot




def full_recording_autoEPG(dic0):
    """
    DESCRIPTION
        this method receives the dictionary of files and produces and then saves the annotation and their time points to file.
    """
    subPath = findPath(dic = dic0, k = dic0.keys()[0])
    for i in dic0:
        peak, annot = autoEPG(k = i, dic = dic0)
        r = (peak, annot)
        pickle.dump(r, open(cwd + subPath + 'autoEPG %d.cPickle' % i, 'wb'))


def goodness(FileName, Peak, Annot):
    """
    DESCRIPTION
        this method evaluates if the Annotation is within the correct range of the peak. 
    """
    df = pd.DataFrame({'e':[0, 0, 0], 'E':[0, 0, 0], 'P':[0, 0, 0], 'R':[0, 0, 0], 'r':[0, 0, 0]}, index = ['Correct', 'Wrong', 'Total'])
    dfx, stamp = raw(int(FileName[-7:-4]))
    del dfx
    for i in range(len(Annot)):
        df[Annot[i]].loc['Total'] += 1
        if (Annot[i] == 'e' or Annot[i] == 'E'):
            if (len(stamp[stamp['t0'] < Peak[i]][stamp[stamp['t0'] < Peak[i]]['t1'] > Peak[i]]) == 0):
                df[Annot[i]].loc['Wrong'] += 1
            else:
                df[Annot[i]].loc['Corrent'] += 1
        elif (Annot[i] == 'P'):
            if (len(stamp[stamp['t1'] < Peak[i]][stamp[stamp['t1'] < Peak[i]]['t2'] > Peak[i]]) == 0):
                df[Annot[i]].loc['Wrong'] += 1
            else:
                df[Annot[i]].loc['Corrent'] += 1
        elif (Annot[i] == 'R' or Annot[i] == 'r'):
            if (len(stamp[stamp['t2'] < Peak[i]][stamp[stamp['t2'] < Peak[i]]['t3'] > Peak[i]]) == 0):
                df[Annot[i]].loc['Wrong'] += 1
            else:
                df[Annot[i]].loc['Corrent'] += 1
        else:
            if (len(stamp[stamp['t3'] < Peak[i]][stamp[stamp['t3'] < Peak[i]]['t0'] > Peak[i]]) == 0):
                df[Annot[i]].loc['Wrong'] += 1
            else:
                df[Annot[i]].loc['Corrent'] += 1
    return df





def insert_pump(t, shift = 0, length = 1.5, k = k, dic = dic, annotate_temp = False):
    """
    DESCRIPTION
        this method finds the 
    """
    global stamp2, coords, t_i
    colors = ['b', 'y', 'g', 'm', 'k', 'w', 'c']
    coords = []
    t_0 = int((t+shift)*10000)
    if (length == 1.5):
        t_1 = int((t+shift+0.5)*10000)
    t_2 = int((t+shift+length)*10000)
    fig = sns.plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print('time: %d'%t)
    t_i = t + 5
    t_i = int(t + 1.9)
    sns.plt.clf()
    if (length==1.5):
        sns.plt.plot(df['time(ms)'][t_0:t_1-1], df['voltage(mV)'][t_0:t_1-1]*(-1), color='b')
        sns.plt.plot(df['time(ms)'][t_1:t_2-1], df['voltage(mV)'][t_1:t_2-1]*(-1), color='y')
    else:
        sns.plt.plot(df['time(ms)'][t_0:t_2], df['voltage(mV)'][t_0:t_2]*(-1), 'ro')
        for color_i in range(int(length)):
            t_0 = int((t+shift+color_i)*10000)
            t_2 = int((t+shift+color_i+1)*10000)
            sns.plt.plot(df['time(ms)'][t_0:t_2-1], df['voltage(mV)'][t_0:t_2-1]*(-1), color=colors[color_i%7])
        if (length - int(length) != 0):
            t_0 = int((t+shift+int(length))*10000)
            t_2 = int((t+shift+length)*10000)
            sns.plt.plot(df['time(ms)'][t_0:t_2-1], df['voltage(mV)'][t_0:t_2-1]*(-1), color=colors[int(length)%7])
    sns.plt.title(dic[k] + str(k))
    print 'Select MAJOR peaks..'
    sns.plt.show()
    fig.canvas.mpl_disconnect(cid)

    if (len(coords) % 4 == 0):
        for i in range(len(coords)/4):
            if (coords[4*i]<coords[4*i+1] and coords[4*i+1]<coords[4*i+2] and coords[4*i+2]<coords[4*i+3]):
                temp = {'e1':coords[4*i], 'e2':0,'E1':0 , 'E2':coords[4*i+1],'R2':0, 'R1':coords[4*i+2],'r1':0, 'r2':coords[4*i+3], 'Pi1':[], 'Pi2':[]}
                temp_idx = len(stamp2)
                stamp2.loc[temp_idx] = temp
                # stamp2['e1'].loc[temp_idx] = coords[4*i]
                # stamp2['E2'].loc[temp_idx] = coords[4*i+1]
                # stamp2['R1'].loc[temp_idx] = coords[4*i+2]
                # stamp2['r2'].loc[temp_idx] = coords[4*i+3]
                print (temp_idx)
                find_minor(k=k, dic=dic, idx=temp_idx, repeat=False)
                if (annotate_temp == True):
                    reverse_annotate(k = k, dic = dic)
                    pump_plot_major(k=k, dic=dic, frame=int(t/10.0)+1)
            else:
                print('Invalid Assignment')
                break




def delete_pump(idx, k = k, dic = dic, annotate_temp = False):
    global stamp2
    s = {}
    s[int(stamp2['e1'].loc[idx]/10.0)] = 0
    s[int(stamp2['r2'].loc[idx]/10.0)] = 0
    stamp2 = stamp2.drop(idx)
    save_stamp2(k=k, dic=dic)
    if (annotate_temp == True):
        reverse_annotate(k = k, dic = dic)
        pump_plot_major(k=k, dic=dic, frame=s.keys())




def find_minor(dic = dic, k=k, idx = -10000, margin = 0.1, repeat = True):
    """
    DESCRIPTION
        this method distinguishes the minor regions.
        Passing a FileName indicates that you have already made a stamp file.
        This function assumes that stamp2 is globally defined.
    """
    global stamp2
    error = False
    if (repeat != True):
        repeat = True
        no_repeat = True
    else:
        no_repeat = False
    subPath = findPath(dic=dic, k=k)
    dfx = make_df(subPath + dic[k])
    last = dfx['time(ms)'][dfx.index[-1]]
    idx_last = stamp2.index[-1]
    if (idx != -10000):
        i = idx
    else:
        i = stamp2[stamp2['r1'].isnull()].index[0]
    while (i <= idx_last and (repeat or error)):
        error = False
        Minor = True
        data_valid = True
        t0 = stamp2['e1'].loc[i]/10000
        t1 = stamp2['E2'].loc[i]/10000
        t2 = stamp2['R1'].loc[i]/10000
        t3 = stamp2['r2'].loc[i]/10000
        print ('index %d/%d' % (i, stamp2.shape[0]))
        temp = find4(t = t0, length = t3 - t0, p1 = t1, p2 = t2, shift = margin, dic=dic, k=k)
        # place values in temp into the three different regions
        if (len(temp) == 1 and typekey == 3):
            print ("Request to quit!")
            break
        elif (len(temp) == 3):
            if (temp[0] == temp[1] and temp[0] == temp[2]):
                print ("Request to quit!")
                break
        # Re-assigns the major peaks
        # reduce the value of the counter to allow for assignment of the minor peaks.
        try:
            if ((temp[-1] == temp[-2] and temp[-1] == temp[-3]) or typekey == 3):
                Minor = False
                if (len(temp) % 4 == 3 or (typekey == 3)):
                    stamp2['e1'].loc[i] = temp[0]
                    stamp2['E2'].loc[i] = temp[1]
                    stamp2['R1'].loc[i] = temp[2]
                    stamp2['r2'].loc[i] = temp[3]
                    print('MAJOR saved to DataFrame')
                    error = True
                else:
                    print('MAJOR Peaks: Invalid Re-assignment! Try Again..')
                    error = True
        # Assign the minor peaks
        except IndexError:
            pass    
        if (Minor):
            print ('Minor Peak')
            d1 = []
            d2 = []
            d3 = []
            data_valid = True
            for j in temp:
                if (j < stamp2['e1'].loc[i] or j > stamp2['r2'].loc[i]):
                    data_valid = False
                    print ("Selection Out of Range")
                    error = True
                    break
                elif (j < stamp2['E2'].loc[i]):
                    d1 += [j]
                elif (stamp2['R1'].loc[i] < j):
                    d3 += [j]
                else:
                    d2 += [j]
            if (len(d1) <= 2 and len(d3) <= 2 and len(d2) % 2 == 0 and data_valid):
                # no e peak
                if (len(d1) == 0):
                    stamp2['e2'].loc[i] = stamp2['e1'].loc[i]
                    stamp2['E1'].loc[i] = stamp2['e1'].loc[i]
                # no plateau region
                elif (len(d1) == 1):    
                    stamp2['e2'].loc[i] = d1[0]
                    stamp2['E1'].loc[i] = d1[0]
                # separate e peak
                else:
                    stamp2['e2'].loc[i] = min(d1)
                    stamp2['E1'].loc[i] = max(d1)
                # no r peak
                if (len(d3) == 0):
                    stamp2['R2'].loc[i] = stamp2['r2'].loc[i]
                    stamp2['r1'].loc[i] = stamp2['r2'].loc[i]
                # no plateau region
                elif (len(d3) == 1):
                    stamp2['R2'].loc[i] = d3[0]
                    stamp2['r1'].loc[i] = d3[0]
                # separate r peak
                else:
                    stamp2['R2'].loc[i] = min(d3)
                    stamp2['r1'].loc[i] = max(d3)
                stamp2['Pi1'].loc[i] = []
                stamp2['Pi2'].loc[i] = []
                while (len(d2) > 0):
                    Pi1 = min(d2)
                    d2.remove(Pi1)
                    stamp2['Pi1'].loc[i] += [Pi1]
                    Pi2 = min(d2)
                    d2.remove(Pi2)
                    stamp2['Pi2'].loc[i] += [Pi2]
                i += 1
            else:
                print('MINOR Peaks: Invalid Input! Try Again..')
                error = True
        if (no_repeat):
            repeat = False
        stamp2.sort_values('e1').to_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Stamps - incomplete.cPickle'))







def save_stamp2(k = k, dic = dic):
    """
    DESCRIPTION
        this method is meant to save stamp2 to the file as excel.
    """
    subPath = findPath(k = k, dic = dic)
    if (os.path.isfile(cwd + subPath + dic[k].replace('.abf', ' Full Stamps - incomplete.cPickle'))):
        prompt = '> '
        print('\'%s\' exists, press \'y\' to overwrite' % (dic[k].replace('.abf', ' Full Stamps - incomplete.cPickle')))
        ans = raw_input(prompt)
        if (ans == 'y'):
            stamp2.to_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Stamps - incomplete.cPickle'))
    else:
        stamp2.to_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Stamps - incomplete.cPickle'))



def backupR2(k = k, dic = dic):
    """
    DESCRIPTION
        this method is creates a backup of the existing "Full Annotation - incomplete.cPickle" to "Full Annotation.cPickle".
    """
    subPath = findPath(k = k, dic = dic)
    stamp_incomplete = pd.read_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Stamps - incomplete.cPickle'))
    if (os.path.isfile(cwd + subPath + dic[k].replace('.abf', ' Full Stamps.cPickle'))):
        prompt = '> '
        print('\'%s\' already exists, press \'y\' to overwrite' % (dic[k].replace('.abf', ' Full Stamps.cPickle')))
        ans = raw_input(prompt)
        if (ans == 'y'):
            stamp_incomplete.to_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Stamps.cPickle'))
    else:
        stamp_incomplete.to_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Stamps.cPickle'))



def loadR(k = k, dic = dic):
    subPath = findPath(k = k, dic = dic)
    if os.path.isfile(cwd + subPath + dic[k].replace('.abf', '.xlsx')):
        return pd.read_excel(cwd + subPath + dic[k].replace('.abf', '.xlsx'))
    else:
        print ("such file doesn't exist")
        return 0


def loadAnnot(k = k, dic = dic):
    subPath = findPath(k = k, dic = dic)
    if os.path.isfile(cwd + subPath + dic[k].replace('.abf', ' Annotation.cPickle')):
        return pickle.load(open(cwd + subPath + dic[k].replace('.abf', ' Annotation.cPickle'), 'rb'))
    else:
        print ("Annotation %d doesn't exist" % k)
        return 0


def loadFullAnnot(k = k, dic = dic):
    subPath = findPath(k = k, dic = dic)
    if os.path.isfile(cwd + subPath + dic[k].replace('.abf', ' Full Annotation.cPickle')):
        return pickle.load(open(cwd + subPath + dic[k].replace('.abf', ' Full Annotation.cPickle'), 'rb'))
    else:
        print ("Full Annotation %d doesn't exist" % k)
        return 0


def loadR2(k = k, dic = dic):
    subPath = findPath(k = k, dic = dic)
    if os.path.isfile(cwd + subPath + dic[k].replace('.abf', ' Full Stamps - incomplete.cPickle')):
        return pd.read_pickle(cwd + subPath + dic[k].replace('.abf', ' Full Stamps - incomplete.cPickle'))
    else:
        print ("Full Stamps %d doesn't exist" % k)
        return 0




if __name__ == '__main__':
    # directories
        ## CTF desktops
    if os.path.isdir('C:\\Users\palikh\Downloads\Python_Scripts\\'):
        cwd = 'C:\\Users\palikh\Downloads\Python_Scripts\\'
        ## laptop Spectre
    elif os.path.isdir('C:\Users\Pouria\Documents\Python Scripts\JD\\'):
        cwd = 'C:\Users\Pouria\Documents\Python Scripts\JD\\'
        ## laptop JW
    elif os.path.isdir('/Downloads/Python_Scripts/'):
        cwd = '/home/hoopoo685/Downloads/Python_Scripts/'
        ## mcgill servers
    elif os.path.isdir('/home/2015/palikh/COMP396/shared/pouriya_motifs/Data/'):
        cwd = '/home/2015/palikh/COMP396/shared/JoeDent/Data'
    typekey = 1
    subPath = findPath(dic=d0, k=d0.values()[0])
    dic = d0
    k = 13
    coords = []
    # df, stamp = raw(k, dic)










