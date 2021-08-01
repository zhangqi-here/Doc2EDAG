import pickle
import json

def default_load_pkl(pkl_file_path, **kwargs):
    with open(pkl_file_path, 'rb') as fin:
        obj = pickle.load(fin, **kwargs)

    return obj

def load_origin_data(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
    return data

def calc_len(sentences, index, offset):
    sentences_len = 0
    # print(len(sentences))
    for i in range(len(sentences)):
        if i < index:
            sentences_len += len(sentences[i]) + 1
    return sentences_len + offset

file = ['dev','test']
model = 'DCFEE-M'
for filename in file:
    origin_data =load_origin_data("/Users/zhangqi/Documents/Doc2EDAG/Data/Doc2EDAG_{}.json".format(filename))
    gold_record_mat = default_load_pkl(
        "/Users/zhangqi/Documents/Doc2EDAG/Exps/run_dee_task/dee_eval.{}.gold_span.{}.100.pkl".format(filename, model))
    pred_record_mat_all = default_load_pkl(
        "/Users/zhangqi/Documents/Doc2EDAG/Exps/run_dee_task/dee_eval.{}.pred_span.{}.100.pkl".format(filename, model))

    ans = []
    print(len(pred_record_mat_all))
    # for iiiii in range(len(pred_record_mat_all)):
    #     for jjjjj in range(5):
    #         if jjjjj != 3 and pred_record_mat_all[iiiii][2][jjjjj] is not None:
    #             print(pred_record_mat_all[iiiii][2])
    for index in range(len(pred_record_mat_all)):
        paragraph = origin_data[index]
        sentences = paragraph[1]['sentences']
        all_sentences = ""
        for sl in sentences:
            # print(sl)
            if (len(sl) != 0):
                all_sentences += sl + "。"

        ans_json = {
            "sentences": all_sentences,
            "events": [

            ]

        }
        term = pred_record_mat_all[index]
        if term is None:
            ans.append(ans_json)
            continue
        ex_idx, pred_event_type_labels, pred_record_mat = term[:3]
        #guid = term[:5]
        span_token_tup_list = term[3].span_token_tup_list
        span_dranges_list = term[3].span_dranges_list
        token_id2span_range = {}
        for token_ids, token_range in zip(span_token_tup_list, span_dranges_list):
            token_id2span_range[token_ids] = token_range

        pred_record_mat = [
            [
                [
                    tuple(arg_tup) if arg_tup is not None else None
                    for arg_tup in pred_record
                ] for pred_record in pred_records
            ] if pred_records is not None else None
            for pred_records in pred_record_mat
        ]
        for events_polarity in range(len(pred_record_mat)):
            event_pol = pred_record_mat[events_polarity]
            if event_pol is not None:
                if events_polarity == 3:
                    polarity = '肯定'
                elif events_polarity == 4:
                    polarity = '否定'
                elif events_polarity == 5:
                    polarity = '可能'
            else:
                continue
            events = event_pol
            if events is None:
                ans.append(ans_json)
                continue
            for i in range(len(events)):
                event = events[i]
                trigger_token_ids = event[0]
                sub_token_ids = event[1]
                obj_token_ids = event[2]
                time_token_ids = event[3]
                loc_token_ids = event[4]

                if trigger_token_ids is not None:
                    trigger_range = token_id2span_range[trigger_token_ids]
                    trigger_text = sentences[trigger_range[0][0]][trigger_range[0][1]: trigger_range[0][2]]
                else:
                    trigger_text = None
                if sub_token_ids is not None:
                    sub_range = token_id2span_range[sub_token_ids]
                    sub_text = sentences[sub_range[0][0]][sub_range[0][1]:sub_range[0][2]] #（0,1，,22）（1,3,44）
                else:
                    sub_text = None
                if obj_token_ids is not None:
                    obj_range = token_id2span_range[obj_token_ids]
                    obj_text = sentences[obj_range[0][0]][obj_range[0][1]:obj_range[0][2]]
                else:
                    obj_text = None
                if time_token_ids is not None:
                    time_range = token_id2span_range[time_token_ids]
                    time_text = sentences[time_range[0][0]][time_range[0][1]:time_range[0][2]]
                else:
                    time_text = None
                if loc_token_ids is not None:
                    loc_range = token_id2span_range[loc_token_ids]
                    loc_text = sentences[loc_range[0][0]][loc_range[0][1]:loc_range[0][2]]
                else:
                    loc_text = None

                tmp_ans = {}
                if trigger_text is not None:
                    tmp_ans["trigger"] = {
                        "text": trigger_text,
                        "length": trigger_range[0][2] - trigger_range[0][1],
                        "offset": calc_len(sentences,trigger_range[0][0],trigger_range[0][1])
                    }
                else:
                    tmp_ans["trigger"] = {
                        "text": None,
                        "length": None,
                        "offset": None
                    }
                tmp_ans["arguments"] = []
                if sub_text is not None:
                    tmp_ans["arguments"].append({
                        'role': 'subject',
                        "text": sub_text,
                        "length": sub_range[0][2] - sub_range[0][1],
                        "offset": calc_len(sentences, sub_range[0][0], sub_range[0][1])
                    })

                if obj_text is not None:
                    tmp_ans["arguments"].append({
                        'role': 'object',
                        "text": obj_text,
                        "length": obj_range[0][2] - obj_range[0][1],
                        "offset": calc_len(sentences, obj_range[0][0], obj_range[0][1])
                    })

                if time_text is not None:
                    tmp_ans["arguments"].append({
                        'role': 'time',
                        "text": time_text,
                        "length": time_range[0][2] - time_range[0][1],
                        "offset": calc_len(sentences, time_range[0][0], time_range[0][1])
                    })

                if loc_text is not None:
                    tmp_ans["arguments"].append({
                        'role': 'loc',
                        "text": loc_text,
                        "length": loc_range[0][2] - loc_range[0][1],
                        "offset": calc_len(sentences, loc_range[0][0], loc_range[0][1])
                    })
                tmp_ans['polarity'] = polarity
                ans_json["events"].append(tmp_ans)
        ans.append(ans_json)

    with open("{}-{}.json".format(model, filename), 'w', encoding='utf-8') as f:
        json.dump(ans, f, ensure_ascii=False)

