import os
import sys
# import data_io
import tools
import jaccard_index as ji
import traceback
DEBUG = True

def main():

    try:
        # get the ref file
        ref_path = sys.argv[1]
        # get the res file
        res_path = sys.argv[2]
        dataset = sys.argv[3]
        # build ref dict
        gt_table, js_table = tools.read_formatted_file_hci(ref_path)
        p_table, _ = tools.read_formatted_file_hci(res_path)
        if dataset == 'ipn':
            Jc = {i+1:[] for i in range(13)}
        if dataset == 'hcigesture':
            Jc = {i:[] for i in range(10)}
        if dataset == 'praxis':
            Jc = {i:[] for i in range(3)}
        for video in p_table:
            ps = p_table[video]
            gts = gt_table[video]
            #帧相同标签计算
            labels = set()
            g_labels = set()
            for seg in gts:
                _, l = seg
                labels.add(l)
                g_labels.add(l)
            for seg in ps:
                _, l = seg
                labels.add(l)
            sum_jsi_v = 0.
            for label in labels:
                jsi_value = ji.Jsi(gts, ps, label)

                Jc[label].append(jsi_value)

                sum_jsi_v += jsi_value
            Js = sum_jsi_v / len(g_labels)
            js_table[video] = Js
        mean_jaccard_index = sum(js_table.values()) / float(len(js_table))
    except Exception as err:
        print(err)
        print(traceback.print_exc())
        return
    # score_result = open(sys.argv[3], 'wb')
    # score_result.write("Accuracy: %0.6f\n" % mean_jaccard_index)
    # score_result.close()
    print('mean Jaccard Index (total)', mean_jaccard_index)
    # print(Jc)
    
    print('mean Jaccard Index (each class)')
    total = 0
    for label in Jc:
        Jc_t = sum(Jc[label])/len(Jc[label])
        total += Jc_t
        print(round(Jc_t, 4), end=' ')
    # print(total/10)
if __name__ == '__main__':
    main()
    # if DEBUG:
    #     data_io.show_io(input_dir, output_dir)
