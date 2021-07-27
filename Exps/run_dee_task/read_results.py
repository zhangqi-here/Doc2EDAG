import pickle

def default_load_pkl(pkl_file_path, **kwargs):
    with open(pkl_file_path, 'rb') as fin:
        obj = pickle.load(fin, **kwargs)

    return obj

gold_record_mat = default_load_pkl("D:/ecnu/Event extraction/Doc2EDAG/Exps/run_dee_task/Output/dee_eval.dev.gold_span.GreedyDec.15.pkl")
pred_record_mat = default_load_pkl("D:/ecnu/Event extraction/Doc2EDAG/Exps/run_dee_task/Output/dee_eval.dev.pred_span.GreedyDec.15.pkl")

print(gold_record_mat[:1])
print(pred_record_mat[:1])

for pred_records in pred_record_mat:
    if pred_records is None:
        print(None)
    print("pred_records: ", pred_records)
    for pred_record in pred_records:
        print("pred_record: ", pred_record)
        # for arg_tup in pred_record:
        #     if arg_tup is None:
        #         print(None)
        #     else:
        #         print(arg_tup)


# pred_record_mat = [
#             [
#                 [
#                     tuple(arg_tup) if arg_tup is not None else None
#                     for arg_tup in pred_record
#                 ] for pred_record in pred_records
#             ] if pred_records is not None else None
#             for pred_records in pred_record_mat[:2]
#         ]
# print(pred_record_mat)
# gold_record_mat = [
#             [
#                 [
#                     tuple(doc_fea.span_token_ids_list[arg_idx]) if arg_idx is not None else None
#                     for arg_idx in event_arg_idxs
#                 ] for event_arg_idxs in event_arg_idxs_objs
#             ] if event_arg_idxs_objs is not None else None
#             for event_arg_idxs_objs in doc_fea.event_arg_idxs_objs_list
#         ]
# pred_record_mat = [
#             [
#                 [
#                     tuple(arg_tup) if arg_tup is not None else None
#                     for arg_tup in pred_record
#                 ] for pred_record in pred_records
#             ] if pred_records is not None else None
#             for pred_records in pred_record_mat
#         ]

# print(pred_record_mat)