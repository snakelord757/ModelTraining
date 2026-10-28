import argparse,json
from model_settings import ROOT
from picnic_inference import Model

def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group();group.add_argument('--base',action='store_true');group.add_argument('--previous',action='store_true');args=p.parse_args()
    adapter=None if args.base else ROOT/'outputs/qwen-picnic-grounded-v1/adapter' if args.previous else ROOT/'outputs/qwen-picnic-grounded-v2/adapter'
    if adapter is not None and not (adapter/'adapter_model.safetensors').is_file():raise SystemExit('Adapter is not ready yet')
    model=Model(adapter)
    from picnic_search_v2 import Search,messages
    search=Search()
    print('Пикник: гибридный поиск по PDF. /exit — выход. Ответы модели требуют проверки по источнику.',flush=True)
    while True:
        try:q=input('\nВопрос: ').strip()
        except (KeyboardInterrupt,EOFError):break
        if q=='/exit':break
        if not q:continue
        hits=search(q)
        try:answer=model.answer(messages(q,hits))
        except ValueError as e:print(str(e));continue
        print(answer['answer'],flush=True)
        print('Найденные страницы:',', '.join(map(str,dict.fromkeys(h['page'] for h in hits))),flush=True)

if __name__=='__main__':main()
