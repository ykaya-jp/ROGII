import json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
slug = "rogii-wellbore-geology-prediction"
ids = [697416, 697400, 694973, 697418, 697431, 697433, 697406, 697857, 698002, 697329, 698562, 697552, 698185, 697507, 698266, 698449, 698282, 697500, 697506, 697300]

def msg_to_dict(m, depth=0):
    d = {}
    for k in ["id","author","authorUserName","postDate","post_date","content","markdown","upvoteCount","upvote_count","editDate","edit_date","parentMessageId","parent_message_id"]:
        v = getattr(m, k, None)
        if v is not None:
            if hasattr(v, "__dict__") and not isinstance(v, (str,int,float,list,dict)):
                # nested
                v = {kk: getattr(v, kk, None) for kk in ["displayName","userName","tier","id"] if getattr(v,kk,None) is not None}
            d[k] = v
    children = getattr(m, "messages", None) or getattr(m, "replies", None) or []
    if children:
        d["replies"] = [msg_to_dict(c, depth+1) for c in children]
    return d

results = {}
for tid in ids:
    try:
        resp = api.competition_list_topic_messages(competition=slug, topic_id=tid, page_size=-1)
        msgs_attr = None
        for k in ["messages","topMessages","top_messages","threadMessages","thread_messages"]:
            v = getattr(resp, k, None)
            if v:
                msgs_attr = v; break
        msgs = msgs_attr or []
        out = [msg_to_dict(m) for m in msgs]
        results[tid] = out
        print(f"topic {tid}: {len(out)} top-level messages")
    except Exception as e:
        print(f"topic {tid}: ERROR {e}")
        results[tid] = {"error": str(e)}

with open("/tmp/rogii-disc/messages_all.json","w") as f:
    json.dump(results, f, default=str, ensure_ascii=False, indent=2)
print("\nSaved to /tmp/rogii-disc/messages_all.json")
