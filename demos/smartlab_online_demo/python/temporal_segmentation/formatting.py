import os

def generate_labels(read_path, save_path):
    f_in = open(read_path, "r")
    f_out = open(save_path, "w+")
    contents = f_in.readlines()
    last_index = 0
    for line in contents:
        if line == '\n':
            continue
        l = line.split()
        index = int(l[0][5:]) + 1
        f_out.writelines([l[1] + "\n" for _ in range(index-last_index)])
        last_index = index
    f_in.close()
    f_out.close()

def label_all_files(dir_in, dir_out):
    paths = os.listdir(dir_in)
    for p in paths:
        route = dir_in + '/' + p
        #print(route)
        if os.path.isfile(route):
            generate_labels(route, dir_out+'/'+p)

def reformat_prediction(path_in, path_out):
    f_in = open(path_in, 'r')
    f_out = open(path_out+'_pred.txt', 'w+')
    contents = f_in.read()
    contents = contents.split('\n')[1].split()
    f_out.writelines([word + '\n' for word in contents])
    f_in.close()
    f_out.close()