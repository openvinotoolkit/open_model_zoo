import numpy as np

from docx import Document
from docx.shared import Mm
from docx.enum.table import WD_TABLE_ALIGNMENT


def mat_inter(box1, box2):
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    if mat_inter(box1, box2):
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / (area1 + area2 - intersection)
        if coincide > 0.05:
            return True
        else:
            return False
    else:
        return False


def remove_redundant_rect(detection_result):
    n = len(detection_result)
    index = [1 for i in range(n)]
    it = list(detection_result.keys())
    for i in range(n - 1):
        for j in range(i + 1, n):
            box1_name = it[i]
            box2_name = it[j]

            box1_cord = detection_result[box1_name]  # x y w h
            box2_cord = detection_result[box2_name]

            box1 = (box1_cord[0], box1_cord[1], box1_cord[0] + box1_cord[2], box1_cord[1] + box1_cord[3])
            box2 = (box2_cord[0], box2_cord[1], box2_cord[0] + box2_cord[2], box2_cord[1] + box2_cord[3])

            flag = solve_coincide(box1, box2)
            if flag:
                target = i if box1_cord[2] * box1_cord[3] > box2_cord[2] * box2_cord[3] else j
                index[target] = 0
    index_np = np.array(index)
    index_it = np.nonzero(index_np)
    it_np = np.array(it)
    result_name = list(it_np[index_it])
    res = {key: value for key, value in detection_result.items() if key in result_name}
    return res


def split_height(ratio_detc_results):
    r = []
    for item in ratio_detc_results.items():
        r.append([item[1][1], item[0]])
    r = sorted(r, key=lambda x: x[0])
    sorted_height = []
    n = len(r)
    i = 0
    while i < n:
        t = r[i][0]
        adj = [r[i][:]]
        i += 1
        while i < n:
            if t - 5 < r[i][0] < t + 5:
                adj.append(r[i][:])
                i += 1
            else:
                break
        sorted_height.append(adj[:])
    sorted_cell_name = []
    for f in sorted_height:
        tmp = []
        for it in f:
            tmp.append(it[1])
        sorted_cell_name.append(tmp)
    return sorted_cell_name


def horizontal_sort(vertical_split, ratio_detc_results):
    sorted_cell = []
    for v in vertical_split:
        img_x = []
        for img in v:
            img_x.append([img, ratio_detc_results[img][0]])
        img_x.sort(key=lambda x: x[1])
        sorted_cell.append(img_x)

    res = [[k[0] for k in it] for it in sorted_cell]

    return res


def to_doc(data, ocr_results, ratio_detc_results, path):
    doc = Document()
    rows_num = len(data)
    cols_num = max(map(len, data))
    table = doc.add_table(rows=rows_num, cols=cols_num)
    table.allow_autofit = False
    table.style = doc.styles['Table Grid']
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    text_data = [[ocr_results[k] for k in it] for it in data]
    # merge cells
    for i in range(rows_num):
        cur_col_num = len(data[i])
        cur_cells = table.rows[i].cells
        index_start = cur_col_num - 1
        for cur_index in range(index_start + 1, cols_num):
            cur_cells[index_start].merge(cur_cells[cur_index])

    i = 0
    for val_row in text_data:
        j = 0
        for val_col in val_row:
            table.rows[i].cells[j].text = val_col
            j += 1
        i += 1

    # set cells size
    cells_size = [[(ratio_detc_results[k][2], ratio_detc_results[k][3]) for k in it] for it in data]

    # set rows height
    for i in range(rows_num):
        cur_height = max(list(map(lambda x: x[1], cells_size[i])))
        table.rows[i].height = Mm(cur_height / 3)
    # set cells width
    for i in range(rows_num):
        cur_col_num = len(data[i])
        for j in range(cur_col_num):
            cur_width = cells_size[i][j][0]
            table.rows[i].cells[j].width = Mm(cur_width / 3)
    doc.save(path)


def save_doc(reco_results, detc_results, path='demo.docx'):
    vertical_split = split_height(detc_results)
    data = horizontal_sort(vertical_split, detc_results)

    to_doc(data, reco_results, detc_results, path)
