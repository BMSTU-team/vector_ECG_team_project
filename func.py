def rename_columns(df):
    # Приводит к правильному виду данные в df:
    new_columns = []
    for column in df.columns:
        new_columns.append(column[:-4])
    df.columns = new_columns
    return df


def function(file="Data VECG\PatientA__Exam_1_0.edf", n_term=3, filt=False, f_sreza=0.5):
    import mne
    import pandas as pd
    from matplotlib import pyplot as plt
    import numpy as np
    import os
    from scipy import signal
    from matplotlib.pyplot import figure
    from catboost import CatBoostClassifier

    data = mne.io.read_raw_edf(file)
    raw_data = data.get_data()
    # you can get the metadata included in the file and a list of all channels:
    info = data.info
    channels = data.ch_names
    fd = 500 # Частота дискретизации
    df = pd.DataFrame(data=raw_data.T,    # values
                index=range(raw_data.shape[1]),    # 1st column as index
                columns=channels)  # 1st row as the column names
    if 'ECG I-Ref' in df.columns:
        df = rename_columns(df)
    Ts = 1/fd
    t = []
    for i in range(raw_data.shape[1]):
        t.append(i*Ts)

    channels = df.columns
    model = CatBoostClassifier()      # parameters not required.
    model.load_model('boosting_model_ECG.cbm')
    sig = np.array(df['ECG I'])
    window = 200
    dataset_check = []
    middles = []
    for i in range(0, len(sig)-window, 6):
        piece = sig[i:i+window] / np.max(np.abs(sig[i:i+window]))
        piece = piece - np.mean(piece)
        middle = (i + i + window) / 2
        middles.append(middle)
        dataset_check.append(piece)
    df_check = pd.DataFrame(dataset_check)
    test_preds = model.predict(df_check, prediction_type="Class")
    peaks = np.where(test_preds > 0)[0]

    # Сделаем временный сигнал, который всегда имеет min значение = 0
    temp_sig = sig
    if min(sig) < 0:
        temp_sig = sig + abs(min(sig))
    if min(sig) > 0:
        temp_sig = sig - abs(min(sig))

    h = max(temp_sig)/1.5  # Выберем только те пики, которые >
    true_peaks = []

    for i in peaks:
        m = int(middles[i])
        if temp_sig[m] > h:
            true_peaks.append(i)

    middles = np.asarray(middles)
    coordinates = middles[true_peaks].astype(np.int64)
    coordinates = np.concatenate((coordinates, max(coordinates)+10000), axis=None)
    final_coord = []
    val_last = 0
    data_points = []
    for val in coordinates:
        if val - val_last > (0.2/Ts):
            if len(data_points)!=0:
                final_coord.append(int(np.array(data_points).mean()))
                data_points = []
        data_points.append(val)
        val_last = val

    if filt != True:
        print('Вот, как ML модель распознала QRS пики:')
        for graph in channels:
            sig = np.array(df[graph])
            figure(figsize=(15, 2), dpi=80)
            plt.plot(sig)
            plt.scatter(final_coord, sig[final_coord], color='red')
            plt.title(graph)
            plt.xlim([0, 5000])
            plt.show()

    if filt == True:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = signal.butter(3, f_sreza, 'hp', fs=fd, output='sos')
            avg = np.mean(sig)
            filtered = signal.sosfilt(sos, sig)
            filtered += avg
            figure(figsize=(10, 3), dpi=80)
    
            plt.plot(t, sig, color='blue')
            plt.plot(t, filtered, color='red')
            plt.title('Фильтрация '+ str(graph))
            plt.legend(["До фильтрации", "После фильтрации"])
            plt.show()
        
            df_new[graph] = pd.Series(filtered)

        df = df_new
        t = []
        for i in range(raw_data.shape[1]):
            t.append(i*Ts)
        sig = np.array(df['ECG I'])
        dataset_check = []
        middles = []
        for i in range(0, len(sig)-window, 6):
            piece = sig[i:i+window] / np.max(np.abs(sig[i:i+window]))
            piece = piece - np.mean(piece)
            middle = (i + i + window) / 2
            middles.append(middle)
            dataset_check.append(piece)
        df_check = pd.DataFrame(dataset_check)
        test_preds = model.predict(df_check, prediction_type="Class")
        peaks = np.where(test_preds > 0)[0]

        # Сделаем временный сигнал, который всегда имеет min значение = 0
        temp_sig = sig
        if min(sig) < 0:
            temp_sig = sig + abs(min(sig))
        if min(sig) > 0:
            temp_sig = sig - abs(min(sig))

        h = max(temp_sig)/1.5  # Выберем только те пики, которые >
        true_peaks = []

        for i in peaks:
            m = int(middles[i])
            if temp_sig[m]>h:
                true_peaks.append(i)


        middles = np.asarray(middles)
        coordinates = middles[true_peaks].astype(np.int64)
        coordinates = np.concatenate((coordinates, max(coordinates)+10000), axis=None)
        final_coord = []
        val_last = 0
        data_points = []
        for val in coordinates:
            if val - val_last > (0.2/Ts):
                if len(data_points)!=0:
                    final_coord.append(int(np.array(data_points).mean()))
                    data_points = []
            data_points.append(val)
            val_last = val

        print('Вот, как ML модель распознала QRS пики после фильтрации сигнала:')
        for graph in channels:
            sig = np.array(df[graph])
            figure(figsize=(15, 2), dpi=80)
            plt.plot(sig)
            plt.scatter(final_coord, sig[final_coord], color='red')
            plt.title(graph)
            plt.xlim([0, 5000])
            plt.show()

    # Подсчет вЭКГ
    i = n_term
    if type(i) == list:
        print(f"Запрошен диапазон с {i[0]} по {i[1]} период включительно")
        fin = i[1]
        beg = i[0]

    else:
        print(f"Запрошен {i} период")
        fin = i
        beg = i
    start = final_coord[beg-1]
    end = final_coord[fin]
    df_term = df.iloc[start:end,:]
    df_row = df.iloc[start:start+1,:]
    df_term = pd.concat([df_term, df_row])
    DI = df_term['ECG I']
    DII = df_term['ECG II']
    V1 = df_term['ECG V1']
    V2 = df_term['ECG V2']
    V3 = df_term['ECG V3']
    V4 = df_term['ECG V4']
    V5 = df_term['ECG V5']
    V6 = df_term['ECG V6']
    df_term['x'] = -(-0.172*V1-0.074*V2+0.122*V3+0.231*V4+0.239*V5+0.194*V6+0.156*DI-0.01*DII)
    df_term['y'] = (0.057*V1-0.019*V2-0.106*V3-0.022*V4+0.041*V5+0.048*V6-0.227*DI+0.887*DII)
    df_term['z'] = -(-0.229*V1-0.31*V2-0.246*V3-0.063*V4+0.055*V5+0.108*V6+0.022*DI+0.102*DII)

    print('Результаты вычисления:')
    plt.figure(figsize=(7, 7), dpi=80)
    plt.plot(df_term.x,df_term.y)
    plt.title('Фронтальная плоскость')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot()
    plt.figure(figsize=(7, 7), dpi=80)
    plt.plot(df_term.y,df_term.z)
    plt.title('Сагитальная плоскость')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.plot()
    plt.figure(figsize=(7, 7), dpi=80)
    plt.plot(df_term.x, df_term.z)
    plt.title('Аксиальная плоскость')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.plot()
    ax = plt.figure(figsize=(10, 10), dpi=80).add_subplot(projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot(df_term.x, df_term.y, df_term.z, label='вЭКГ')
    ax.legend()

    plt.show()




def vECG(file="Data VECG\PatientA__Exam_1_0.edf", n_term=3, filt=False, f_sreza=0.5):
    import mne
    import pandas as pd
    from matplotlib import pyplot as plt
    import numpy as np
    import os
    from scipy import signal
    from matplotlib.pyplot import figure
    from catboost import CatBoostClassifier

    data = mne.io.read_raw_edf(file)
    raw_data = data.get_data()
    # you can get the metadata included in the file and a list of all channels:
    info = data.info
    channels = data.ch_names
    fd = 500 # Частота дискретизации
    df = pd.DataFrame(data=raw_data.T,    # values
                index=range(raw_data.shape[1]),    # 1st column as index
                columns=channels)  # 1st row as the column names
    if 'ECG I-Ref' in df.columns:
        df = rename_columns(df)
    Ts = 1/fd
    t = []
    for i in range(raw_data.shape[1]):
        t.append(i*Ts)

    channels = df.columns
    model = CatBoostClassifier()      # parameters not required.
    model.load_model('boosting_model_ECG.cbm')
    sig = np.array(df['ECG I'])
    window = 200
    dataset_check = []
    middles = []
    for i in range(0, len(sig)-window, 6):
        piece = sig[i:i+window] / np.max(np.abs(sig[i:i+window]))
        piece = piece - np.mean(piece)
        middle = (i + i + window) / 2
        middles.append(middle)
        dataset_check.append(piece)
    df_check = pd.DataFrame(dataset_check)
    test_preds = model.predict(df_check, prediction_type="Class")
    peaks = np.where(test_preds > 0)[0]

    # Сделаем временный сигнал, который всегда имеет min значение = 0
    temp_sig = sig
    if min(sig) < 0:
        temp_sig = sig + abs(min(sig))
    if min(sig) > 0:
        temp_sig = sig - abs(min(sig))

    h = max(temp_sig)/1.5  # Выберем только те пики, которые >
    true_peaks = []

    for i in peaks:
        m = int(middles[i])
        if temp_sig[m] > h:
            true_peaks.append(i)


    middles = np.asarray(middles)
    coordinates = middles[true_peaks].astype(np.int64)
    coordinates = np.concatenate((coordinates, max(coordinates)+10000), axis=None)
    final_coord = []
    val_last = 0
    data_points = []
    for val in coordinates:
        if val - val_last > (0.2/Ts):
            if len(data_points)!=0:
                final_coord.append(int(np.array(data_points).mean()))
                data_points = []
        data_points.append(val)
        val_last = val


    if filt == True:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = signal.butter(3, f_sreza, 'hp', fs=fd, output='sos')
            avg = np.mean(sig)
            filtered = signal.sosfilt(sos, sig)
            filtered += avg       
            df_new[graph] = pd.Series(filtered)

        df = df_new
        t = []
        for i in range(raw_data.shape[1]):
            t.append(i*Ts)
        sig = np.array(df['ECG I'])
        dataset_check = []
        middles = []
        for i in range(0, len(sig)-window, 6):
            piece = sig[i:i+window] / np.max(np.abs(sig[i:i+window]))
            piece = piece - np.mean(piece)
            middle = (i + i + window) / 2
            middles.append(middle)
            dataset_check.append(piece)
        df_check = pd.DataFrame(dataset_check)
        test_preds = model.predict(df_check, prediction_type="Class")
        peaks = np.where(test_preds > 0)[0]

        # Сделаем временный сигнал, который всегда имеет min значение = 0
        temp_sig = sig
        if min(sig) < 0:
            temp_sig = sig + abs(min(sig))
        if min(sig) > 0:
            temp_sig = sig - abs(min(sig))

        h = max(temp_sig)/1.5  # Выберем только те пики, которые >
        true_peaks = []

        for i in peaks:
            m = int(middles[i])
            if temp_sig[m]>h:
                true_peaks.append(i)


        middles = np.asarray(middles)
        coordinates = middles[true_peaks].astype(np.int64)
        coordinates = np.concatenate((coordinates, max(coordinates)+10000), axis=None)
        final_coord = []
        val_last = 0
        data_points = []
        for val in coordinates:
            if val - val_last > (0.2/Ts):
                if len(data_points)!=0:
                    final_coord.append(int(np.array(data_points).mean()))
                    data_points = []
            data_points.append(val)
            val_last = val

    # Подсчет вЭКГ
    i = n_term
    if type(i) == list:
        print(f"Запрошен диапазон с {i[0]} по {i[1]} период включительно")
        fin = i[1]
        beg = i[0]

    else:
        print(f"Запрошен {i} период")
        fin = i
        beg = i
    start = final_coord[beg-1]
    end = final_coord[fin]
    df_term = df.iloc[start:end,:]
    df_row = df.iloc[start:start+1,:]
    df_term = pd.concat([df_term, df_row])
    DI = df_term['ECG I']
    DII = df_term['ECG II']
    V1 = df_term['ECG V1']
    V2 = df_term['ECG V2']
    V3 = df_term['ECG V3']
    V4 = df_term['ECG V4']
    V5 = df_term['ECG V5']
    V6 = df_term['ECG V6']
    df_term['x'] = -(-0.172*V1-0.074*V2+0.122*V3+0.231*V4+0.239*V5+0.194*V6+0.156*DI-0.01*DII)
    df_term['y'] = (0.057*V1-0.019*V2-0.106*V3-0.022*V4+0.041*V5+0.048*V6-0.227*DI+0.887*DII)
    df_term['z'] = -(-0.229*V1-0.31*V2-0.246*V3-0.063*V4+0.055*V5+0.108*V6+0.022*DI+0.102*DII)

    print('Результаты вычисления:')
    fig, axs = plt.subplots(1,3,figsize=(15,3.7))
    axs[0].plot(df_term.x, df_term.y)
    axs[0].set_title('Фронтальная плоскость')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')

    axs[1].plot(df_term.y, df_term.z)
    axs[1].set_title('Сагитальная плоскость')
    axs[1].set_xlabel('Y')
    axs[1].set_ylabel('Z')

    axs[2].plot(df_term.x, df_term.z)
    axs[2].set_title('Аксиальная плоскость')
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Z')

    ax = plt.figure(figsize=(10, 10), dpi=80).add_subplot(projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot(df_term.x, df_term.y, df_term.z, label='вЭКГ')
    ax.legend()

    plt.show()