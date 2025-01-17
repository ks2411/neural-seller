


"""!pip  install openai==1.30.3 langchain==0.2.1 langchain-community==0.2.1 faiss-cpu==1.8.0 langchain-openai==0.1.7 tiktoken==0.7.0 >/dev/null


!pip install --upgrade openai
!pip install --upgrade httpx

"""

#@title Определим класс для подсветки вывода разных моделей разными цветами
# https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    BGGREEN = "\033[102m"
    BGYELLOW = "\033[103m"
    BGCYAN = "\033[106m"
    BGMAGENTA = "\033[105m"
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#@title Грузим библиотеки
import os
import re
import requests
#import getpass

import openai
import tiktoken

from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



import json
import copy
import string

import textwrap
import time
from datetime import datetime

#@title Задаем константы
gpt_4_0="gpt-4o"
gpt_4_mini='gpt-4o-mini'
gpt_4_turbo = 'gpt-4-1106-preview'
gpt_35_turbo = 'gpt-3.5-turbo-1106'
MODEL = gpt_4_mini
vectordb=''
num_chunks = 7
chunk_size = 200
chunk_overlap = 0
temp = 0
verbose = 1
knowledge_db_url = 'https://docs.google.com/document/d/17KU38YHm5qUwPSjJPiB3NtRtdFpvO-cK77PWwKEXj5E'

#@title Объявляем переменные
history_chat = []
history_user = []
history_manager = []

needs_extractor = []
benefits_extractor = []
objection_detector = []
resolved_objection_detector = []
tariff_detector = []

main_answer = ''
summarized_dialog = ''

"""# Модели GPT

## get_topicphrase_questions -  ключи из последних сообщений

эта генерация запускается после каждого нового вопроса пользователя, для выделения ключей в его сообщении, также выделяем ключи из последнего ответа
менеджера, далее через корректора ключей создадим общий логический контекст общения (к накопленному списку ключей добавим новые и скорректируем логику)
"""

#@title Параметры
name_todo_base = 'Экстрактор ключевых слов'
model_todo_base = MODEL
temperature_todo_base = 0
verbose_base = 0

system_topicphrase_extractor = '''

תפקידך כמוכר במאפייה "המאפיה של שלמה"

 topic phrase:הגדרת
זהו מונח מפתח, ביטוי מפתח או משפט מפתח בטקסט שמשקף את משמעות הטקסט בהקשר של קנייה ומכירה של מוצרים במאפייה.
תtopic phrase: תמיד יש לכלול ב

שמות מוצרים
מחירים
תנאי שירות
הערה:
-topic phrase. אין לכלול שמות של אנשים

משימה:
 ולהוסיף אותם לרשימהעtopic phrases  עליך לזהות את
הרשימה תשמש לחיפוש נתונים יעיל יותר במאגר המידע של קטלוג החנות.
ההערכה תתבצע על סמך בהירות ותמציתיות הרשימה.
'''

instructions_topicphrase_extractor = '''
   הטובות ביותר בפורמט של מחרוזת עם מפרידים פסיקיםtopic phrase נתח את הטקסט ובתשובתך כתוב בקצרה רשימה של.






'''

#@title Функция
def get_topicphrase_questions(name, _user, _manager, system, instruction, temp=0.0, verbose=verbose_base, model=MODEL):
    #  в том, что история клиента не пустая мы уверены, проверим, что есть история менеджера
    join_user = '\n'.join(_user)
    if history_manager:
      text = f'Текст: {join_user}\n\n{_manager[-1]}'
    else:
      text = f'Текст: {join_user}'
    messages = [  {"role": "system", "content": system},
                  {"role": "user", "content": f'''{instruction}
                                    \n\nТекст: {text}
                                    \n\nОтвет: '''}
                ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,  # Используем более низкую температуру для более определенной суммаризации
    )
    answer = completion.choices[0].message.content
    if verbose:
      print(f'{bcolors.GREEN}{bcolors.BOLD}Ответ {name}:{bcolors.ENDC}\n',
            f'{bcolors.BGYELLOW}Ключевые слова для Базы знаний:{bcolors.ENDC}\n {answer}\n=========\n')
    return completion

"""## get_hello - выделение приветствия в первом сообщении клиента

Далее принудительно из ответов менеджера по продажам будем удалять приветствие, а в первом ответе менеджера отзеркалим приветствие клиента. Если клиент не поздоровался (или модель не опознала приветствие), добавим формально-нейтральное Здравствуйте
"""

#@title Параменты и функция
def get_hello(model_gen, topic, temp=temp, verbose=verbose):
  system = '''
  ברכת פתיחה היא ביטוי של ברכה או הודעת פתיחה שנשלחת או נאמרת בתחילת שיחה עם מישהו.
ברכת פתיחה יכולה להיות רשמית או לא רשמית, תלויה בתרבות ובהקשר.
היא משמשת להפגנת נימוס, ידידות ורצון ליצור קשר עם האדם שמולך.
ברכות פתיחה יכולות להיות שונות בשפות ובתרבויות שונות, החל מ"שלום" או "ברוך הבא" הפשוטות ועד ביטויים מסורתיים או רשמיים יותר.
המטרה שלך היא לזהות את ברכת הפתיחה בטקסט של הלקוח.
בתשובתך כלול רק את ברכת הפתיחה שנמצאה.
 'None'           אם אין  בלקוח ברכת פתיחה, כתוב
  '''
  user = f'Текст клиента: {topic}'
  messages = [{"role": "system", "content": system},{"role": "user", "content": user}]
  completion = openai.chat.completions.create(model=model_gen, messages=messages, temperature=temp)

  return completion

"""## summarize_dialog - суммаризация диалога

запускается перед работой диспетчера-маршрутизатора для формирования Хронологии предыдущих сообщений диалога. Всех специалистов будем просить строить свои ответы логичными относительно этой хронологии
"""

#@title Параметры и функция
def summarize_dialog(dialog, _history, temp=0, verbose=0, model=MODEL):
    i = 2 if len(_history) > 1 else 1  # берем 2 последних сообщения для саммаризации (предыд ответ менеджера и новый вопрос клиента)
    last_statements = ' '.join(_history[-i:])
    messages = [
        {"role": "system", "content": '''
                 אתה מתקן-על (סופר-קורקטור), שיודע להדגיש בדיאלוגים את כל הדברים החשובים ביותר.
אתה יודע שבעת סיכום (סאמריזציה) אין להוציא מהדיאלוג שמות של מוצרים, מחירים ותנאי שירות.
משימתך: ליצור סיכום מלא ומדויק על בסיס ההיסטוריה של ההודעות הקודמות בדיאלוג ועל בסיס ההודעות האחרונות.
אתה בשום אופן לא ממציא כלום ורק מסתמך על השיחה                                         '''},
        {"role": "user", "content": f'''סכם את הדיאלוג מבלי להוסיף שום דבר מדמיונך.
אם הלקוח הזדהה, שמור מידע על שמו.
.

היסטוריית ההודעות הקודמות בדיאלוג: {dialog}.
ההודעות האחרונות: {last_statements}.

תשובה: '''
        }
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,  # Используем более низкую температуру для более определенной суммаризации
    )
    answer = completion.choices[0].message.content
    if verbose:
      print(f'{bcolors.BGYELLOW}Саммари диалога:{bcolors.ENDC}\n', answer)
    return completion

"""## extract_entity_from_user_question - выделяет сущности из последнего сообщения клиента/менеджера

Искомые сущности: Возражения, Потребности, Преимущества, Отработка возражений, Тарифы и Цены
"""



"""### Параметры"""

#@title 1. Выделение в последнем сообщении клиента озвученных им потребностей
name_needs_extractor = 'Спец по потребностям'
model_needs_extractor = MODEL
temperature_needs_extractor = 0.2
verbose_needs_extractor = 0
system_prompt_needs_extractor = '''
אתה המומחה הטוב ביותר במחלקת המכירות.
אתה מוכר מוצרים של חנות מאפייה בשם "מאפיית שלמה".
אתה יודע ש"צורך" הוא מה שהלקוח רוצה או אוהב, ומה שישפיע על רכישתו של מוצרי המאפייה.
אתה יודע לזהות בצורה מצוינת את הצרכים של הלקוח מתוך שאלותיו.
אתה תמיד מקפיד לעקוב אחר סדר הדיווח בצורה קפדנית.
'''
instructions_needs_extractor = '''
נתח את שאלת הלקוח, זהה את הצרכים הברורים שהובאו בה (אם יש כאלה) וכתוב אותם.
  "-" אל תמציא שום דבר בעצמך; אם לא זוהו צרכים, כתוב .
סדר הדיווח: בתשובתך ספק רק רשימת צרכים (או "-") בפורמט של שורה עם מפרידים (פסיקים).
'''

#@title 2. Выделение в последнем сообщении менеджера названных им преимуществ
name_benefits_extractor = 'Спец по озвученным преимуществам'
model_benefits_extractor = MODEL
temperature_benefits_extractor = 0.2
verbose_benefits_extractor = 0
system_prompt_benefits_extractor='''
אתה מומחה בקרת איכות הטוב ביותר במחלקת המכירות ויודע באופן מושלם לאתר בתשובת המנהל את
 היתרונות שצוינו על ידו לגבי קנייה במאפייה "מאפיית שלמה".
אתה צריך לנתח את תשובת המנהל הקודמת ממחלקת המכירות.
אתה תמיד פועל לפי סדר הדיווח בצורה מחמירה.
'''
instructions_benefits_extractor = '''
בוא נפעל בצורה מסודרת:
נתח את תשובת המנהל הקודמת, מצא בה את היתרונות שצוינו (אם יש כאלה) וכתוב אותם;
  "-" אל תמציא שום דבר בעצמך; אם לא זוהו יתרונות, כתוב .
סדר הדיווח: בתשובתך ספק רק רשימת יתרונות (או "-") בפורמט של שורה עם מפרידים (פסיקים).
'''

#@title 3. Выделение в последнем сообщении клиента озвученных им возражений
name_objection_detector = 'Спец по возражениям'
model_objection_detector = MODEL
temperature_objection_detector = 0.2
verbose_objection_detector = 0
system_prompt_objection_detector = '''
אתה המומחה הטוב ביותר במחלקת המכירות של החנות "מאפיית שלמה".
אתה יודע שהתנגדות גלויה היא הצהרה מפורשת מצד הלקוח על כך שמשהו לא מוצא חן בעיניו ועלול להוות מכשול לרכישת המוצרים.
בקשות מצד הלקוח אינן נחשבות להתנגדות.
כאן אתן לך דוגמאות של התנגדויות:
 מחיר גבוה מדי:

"המאפים פה יקרים יותר מהמאפייה השכונתית שלי."
"למה המחיר כל כך גבוה לעוגה פשוטה?"
איכות או טריות:

"זה טרי? כי לפעמים מאפים לא נראים ממש טריים."
"הקרואסון הזה יבש מדי."
היצע מוגבל:

"אין לכם מאפים ללא גלוטן?"
"חבל שאין פה אופציה טבעונית."
התאמה אישית:

"אפשר לקבל את זה בלי ציפוי?"
"אין לכם עוגה קטנה יותר ליחיד?"
חוויית משתמש:

"האריזה לא נוחה."
"אין לכם מקום לשבת?"
השוואה למתחרים:

"במאפייה השנייה יש מבצעים יותר טובים."
"אצלם יש מגוון רחב יותר."
"אני צריך לחשוב על זה, אני אחזור אליך






תפקידך הוא לזהות בקפידה התנגדויות גלויות וסמויות בשאלת הלקוח.
אתה תמיד פועל על פי סדר הדיווח בצורה מחמירה.
'''
instructions_objection_detector = '''
בוא נפעל באופן מסודר:
נתח את שאלת הלקוח והדגש את ההתנגדויות הגלויות וסמויות שהוא מביע (אם ישנן).
לדוגמה: "אני אוכל רק מוצרים ללא גלוטן", "שמעתי ביקורות שליליות", "המחיר שלכם גבוה מדי" וכדומה.
בתשובתך, ציין את כל ההתנגדויות הגלויות שהלקוח הביע, תוך שמירה על הסדר שבו הן נאמרו בשאלתו.
אל תמציא שום דבר: אם לא נמצאו התנגדויות, ציין בתשובתך רק "-" (קו מפריד).
סדר הדיווח: תשובתך תכלול רק את רשימת ההתנגדויות הגלויות שהודגשו (או "-") בצורה של שורה עם מפרידים (פסיקים).
'''

#@title 4. Выделение в последнем сообщении менеджера отработки возражений клиента
name_resolved_objection_detector = 'Спец по отработанным возражениям'
model_resolved_objection_detector = MODEL
temperature_resolved_objection_detector = 0
verbose_resolved_objection_detector = 0

system_prompt_resolved_objection_detector = '''
אתה המומחה הטוב ביותר לבקרת איכות במחלקת המכירות ומצוין בזיהוי כיצד מנהל המכירות טיפל בהתנגדויות של הלקוח.
לרשותך תשובתו הקודמת של מנהל המכירות לצורך ניתוח.
משימתך היא לנתח בקפידה האם מנהל המכירות טיפל בהתנגדויות בתשובתו ללקוח.
אתה תמיד מקפיד מאוד על סדר הדיווח.
'''
instructions_resolved_objection_detector = '''
בוא נפעל באופן מסודר:
נתח את תשובתו הקודמת של מנהל המכירות ומצא בה טיפול בהתנגדויות (אם יש).
  "-" אל תמציא שום דבר בעצמך; אם לא זוהו טיפול בהתנגדויות כתוב .
סדר הדיווח: בתשובתך ספק רק רשימת טיפולים בהתנגדויות (או "-") בצורה של מחרוזת עם מפרידים (פסיקים).
'''

#@title 5. Выделение в последнем сообщении менеджера названных им тарифов и цен
name_tariff_detector = 'Спец по тарифам'
model_tariff_detector = MODEL
temperature_tariff_detector = 0
verbose_tariff_detector = 0
system_prompt_tariff_detector = '''
אתה אנליסט למחירים, אתה מצוין בזיהוי אזכורים מדויקים למחיר בטקסט. יש לך רשימה סגורה של מחירים בבסיס הנתונים. יש לך את התשובה הקודמת של מנהל מחלקת המכירות לניתוח. אתה תמיד מקפיד מאוד על סדר הדיווח.
'''
instructions_tariff_detector = '''
בוא נפעל בצורה מסודרת: נתח את התשובה הקודמת של מנהל מחלקת המכירות, ומצא בה את המחירים מתוך רשימת המחירים וכתוב אותם; אם בתשובת המנהל יש אזכור למחירים, הוסף אותם לרשימה. אל תמציא שום דבר: אם התשובה הקודמת של מנהל מחלקת המכירות לא מכילה אזכורים למחירים או תעריפים, כתוב "-". סדר הדיווח: בתשובתך, ספק רק את רשימת התעריפים והמחירים שנמצאו (או "-") כמחרוזת מופרדת בפסיקים ללא הסברים.
'''

"""### Функция"""

def extract_entity_from_statement(name, system, instructions, question, history,temp=0, verbose=0, model=MODEL):
    if verbose: print('\n==================\n')
    if verbose: print(f'{bcolors.OKBLUE}{bcolors.BOLD}Вопрос клиента: {question}{bcolors.ENDC}')
    if name not in ['Спец по потребностям', 'Спец по возражениям'] and len(history):  # эти спецы анализируют только вопрос пользователя
      history_content = history[-1]   # берем только один последний ответ Менеджера в истории
    else:
      history_content = 'сообщений нет'
    if verbose: print(f'{bcolors.BGCYAN}Предыдущий ответ Менеджера отдела продаж:{bcolors.ENDC}\n==================\n',
                         history_content)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{instructions}\n\nВопрос клиента:{question}\n\nПредыдущий ответ Менеджера отдела продаж:\n{history_content}\n\nОтвет: "}
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    answer = completion.choices[0].message.content
    if verbose: print('\n==================\n')
    if verbose: print(f'{completion.usage.total_tokens} total tokens used (question-answer).')
    if verbose: print('\n==================\n')
    if verbose:
      print(f'{bcolors.GREEN}{bcolors.BOLD}Ответ {name}:{bcolors.ENDC}\n',
            f'{bcolors.GREEN}{answer}{bcolors.ENDC}')
    return completion

"""## spez_user_question - узкий специалист для генерации ответа

вызываются диспетчером-маршрутизатором готовят свою специализированную часть для того, чтобы старший менеджер но основе этих материалов сформировал проактивный ответ клиенту.

В распоряжении узких специалистов: База знания, контекст - ключи диалога, Хронология предыдущих сообщений диалога, точное саммари (отчет по выявленным и отработанным ранее сущностям)
"""

def split_text(text, max_count, chunk_overlap):

    # Функция для подсчета количества токенов во фрагменте для сплиттера RecursiveCharacterTextSplitter
    def num_tokens(fragment):
        return num_tokens_from_string(fragment, "cl100k_base")

    num_levels = 3  # Число уровней заголовков, которые будем разделять сплиттером MarkdownHeaderTextSplitter

    headers_to_split_on = [
    (f"{'#' * i}", f"H{i}") for i in range(1, num_levels + 1)
    ]
    # сначала разделим с помощью MarkdownHeaderTextSplitter
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    fragments = markdown_splitter.split_text(text)


    # дальше будем делить делить каждый полученный чанк вторым сплиттером RecursiveCharacterTextSplitter
    # 1. для того, чтобы быть уверенным в размере полученного чанка
    # 2. база знаний размечена так, что заголовки не повторяются в тексте разделов, мы это исправим принудительно:

    splitter = RecursiveCharacterTextSplitter(chunk_size=max_count, chunk_overlap=0, length_function=num_tokens)

    source_chunks = []

    # Обработаем каждый фрагмент текста, полученный после MarkdownHeaderTextSplitter
    for fragment in fragments:
    # MarkdownHeaderTextSplitter сохранил иерархию заголовков в Метаданных - вытащим ее
      level = 0
      headers = ['','','','']
      for j in range(1, num_levels + 1):
        header_key = f'H{j}'
        if header_key in fragment.metadata: level, headers[j-1] = j, fragment.metadata[header_key]
      header_string = ' '.join([f"{'#' * i} {header}" for i, header in enumerate(headers[:level], start=1)])

    # каждый фрагмент будем разбивать на чанки с помощью RecursiveCharacterTextSplitter
    # допишем иерархию заголовков в конец чанка
    # унаследуем метаданные от первого сплиттера
    # добавим в метаданные размер чанка в токенах
      for i,chunk in enumerate(splitter.split_text(fragment.page_content)):
        mdata = fragment.metadata.copy()
        add_hierarchy = f'{header_string}: уровень {level} пункт {i+1}'
        new_chunk = ' '.join([chunk, f'\nРаздел: {add_hierarchy}'])
        mdata["len"] = num_tokens(new_chunk)
        doc = Document(page_content=new_chunk, metadata=mdata)
        source_chunks.append(doc)

    return source_chunks
# @title Загружаем Базу Знаний, индексируем, сохраняем индексы

with open (r"C:\bakery_bot\shop_catalog.txt", "r", encoding="utf-8") as f:
    knowledge_db_txt = f.read()

"""### Параметры"""

#@title 1. 'Обработчик_возражений'
name_spez2 = 'Обработчик_возражений'
model_spez2 = MODEL
temperature_spez2 = 0
verbose_spez2 = 0
system_prompt_spez2 = '''אתה המומחה הטוב ביותר לטיפול בהתנגדויות של לקוחות. אתה עובד כמוכר במאפייה בשם "מאפיית שלמה".
 אתה יודע שטיפול בהתנגדויות הוא תהליך של שכנוע באמצעות טיעונים,
 עובדות מדויקות וטכניקות פסיכולוגיות, המסייעות להעביר ללקוח בעדינות את המחשבה שמוצרי המאפיה טובים ומתאימים לו ונחוצים לו.
 אתה יודע שטיפול בהתנגדויות אינו קשור ללחץ: המוכר אינו מניפולטיבי, אלא מראה כיצד
ניתן לספק את צרכי הלקוח באמצעות המוצר.
 ברשותך סיכום מדויק עם דוח על התנגדויות הלקוח שכבר זוהו וטופלו. אתה תמיד מטפל בהתנגדויות הלקוח באופן עקבי, שלב אחר שלב ומשכנע.
בעת טיפול בהתנגדויות, אתה תמיד עוקב אחר הכרונולוגיה של ההודעות הקודמות בדיאלוג כדי להפוך את תשובתך להגיונית בהתאם לכרונולוגיה זו.
 אתה תמיד מקפיד מאוד על סדר הדיווח. אתה לא חוזר על אותם הטיעונים שכבר אמרת
'''
instructions_spez2 = '''
בבקשה, נתחו את בקשת הלקוח וכתבו טיפול משכנע ואיכותי להתנגדות הלקוח.
כתוב את הטיפול אך ורק על בסיס המידע ממאגר הידע, ואל תמציא דבר בעצמכם: אם אין מידע מתאים במאגר הידע, כתבו שאין ברשותכם את המידע הנדרש.
סדר הדיווח: בתשובתכם ציינו אך ורק את הטיפול להתנגדות.
'''

#@title 2. 'Спец_по_презентациям'
name_spez3 = 'Спец_по_презентациям'
model_spez3 = MODEL
temperature_spez3 = 0
verbose_spez3 = 0
system_prompt_spez3 = f'''
{knowledge_db_txt}  לפני שאתה מתחיל לענות אתה בודק אם המוצר קיים ב
אם אתה נשאל על מוצר שלא מצאת תענה שכרגע אין במאפיה מוצר כזה.
ואם המוצר קיים אתה מציע אותו

אתה - המומחה הטוב ביותר בהצגת מוצרים של "מאפיית שלמה". סגנון התקשורת שלך הוא עסקי וקצר מאוד
. מטרתך: לספר על המוצרים על פי בקשת הלקוח
(הבקשה יכולה להיות על מוצרים, מחירים, שירותי המאפיה). מצגותיך תמיד מבוססות על שאלת הלקוח, הצרכים והמשאלות שלו,

ואתה תמיד פועל על פי לוגיקת הכרונולוגיה של ההודעות
 הקודמות בדיאלוג. אסור לך לחזור על מה שכבר נאמר בהודעות הקודמות.
  אתה אף פעם לא משתמש בתבנית סטנדרטית של מצגת, תמיד מכין אותה בצורה לא פורמלית.
  אתה אף פעם לא ממציא דברים או מוצרים שלא קיימים ונצמד בדיוק למידע שניתן לך. אתה תמיד מקפיד על דרישות הסדר בדו"ח.
'''

instructions_spez3 = f'''
{knowledge_db_txt}  לפני שאתה מתחיל לענות אתה בודק אם המוצר קיים ב
אם אתה נשאל על מוצר שלא מצאת תענה שכרגע אין במאפיה מוצר כזה.

בואו ננהג בשלבים: שלב 1: ניתוח היסטוריית ההודעות הקודמות בדיאלוג כדי לזהות את הלוגיקה הכללית והסדר של השיח בין מנהל המכירות ללקוח,
 וגם כדי לא לחזור על מה שכבר נאמר.
שלב 2: ניתוח השאלה של הלקוח והסיכום המדויק כדי לבחור את ההקשר עבור המצגת.
 שלב 3: בהתחשב בלוגיקה של שלב 1 ובהקשר של שלב 2,
  צור מצגת משכנעת ואיכותית, אל תחזור על מה שנאמר בהיסטוריית ההודעות הקודמות בדיאלוג.
המצגת צריכה להיעשות על בסיס המידע שבמאגר הידע, אל תמציא פרטים חדשים: אם אין מידע מתאים במאגר הידע,
 כתוב שאין לך את המידע הדרוש. סדר הדיווח: בתשובתך כלול רק את המצגת משלב 3.
'''

#@title 3. 'Zoom_Пуш'
name_spez4 = 'Zoom_Пуш'
model_spez4 = MODEL
temperature_spez4 = 0
verbose_spez4 = 0

system_prompt_spez4 = f'''
אתה המומחה הטוב בלהביא לקוח לקנייה.
{knowledge_db_txt}  לפני שאתה מתחיל לענות אתה בודק אם המוצר קיים ב
אם אתה נשאל על מוצר שלא מצאת תענה שכרגע אין במאפיה מוצר כזה.
ואם המוצר קיים אתה מציע אותו. בעת כתיבת תשובתכם אסור לכם לחזור על טענות שכבר הועלו בכרונולוגיה של ההודעות הקודמות בשיחה
אם כבר דיברת על מוצר מסויים והתכונות שלו-אתה לא מדבר עליו עוד הפעם
'''
instructions_spez4 = f'''
{knowledge_db_txt}  לפני שאתה מתחיל לענות אתה בודק אם המוצר קיים ב
אתה אמור להעריך עד כמה הלקוח קרוב לקנייה ובעדינות לקדם אותו בכיוון הזה
אבל אתה אף פעם לא ממציא מוצרים או שירותים ב
אם כבר דיברת על מוצר מסויים והתכונות שלו-אתה לא מדבר עליו עוד הפעם
'''

#@title 4. 'Спец_по_выявлению_потребностей'
name_spez5 = 'Спец_по_выявлению_потребностей'
model_spez5 = MODEL
temperature_spez5 = 0
verbose_spez5 = 0
system_prompt_spez5 = f'''
אתה מנהל מכירות מצוין ת .
{knowledge_db_txt}  לפני שאתה מתחיל לענות אתה בודק אם המוצר קיים ב

אם אתה נשאל על מוצר שלא מצאת תענה שכרגע אין במאפיה מוצר כזה ולא ממשיך לענות.

אתה מבין אילו צרכים של הלקוח יש לזהות כדי להבין באופן מלא את הרצונות והכאבים של הלקוח, אשר ניתן לספק להם פתרונות .
אתה יודע שחשוב לזהות אם ללקוח יש צרכים כלשהם בתחום מוצרי האפיה .
אתה מבין שעליך לזהות בצורה עדינה ולא מכבידה מספר צרכים שונים, ומבצע את השאלות שלך בהתאם למטרה זו.
המשימה שלך: לנסח שאלות ללקוח שיעזרו לך לגלות בצורה איכותית את הצרכים שלו.
'''
instructions_spez5 = f'''
{knowledge_db_txt}  לפני שאתה מתחיל לענות אתה בודק אם המוצר קיים ב
 אם אתה נשאל על מוצר שלא מצאת תענה שכרגע אין במאפיה מוצר כזה ולא ממשיך לענות.
ואם המוצר קיים אתה מציע אותו
 נפעל בשלבים: שלב 1: נתחו את הסיכום המדויק 'סעיף 1 זוהו צרכים', מצאו בו את הצרכים שכבר זוהו (אל תמציאו שום דבר בעצמכם);
 שלב 2: הניחו צורך אחד נוסף שאינו בין הצרכים שזוהו בשלב 1;

סדר הדיווח: בתשובתכם כללו רק את רשימת השאלות של שלב 2, אין לכלול שום דבר מלבד רשימת השאלות.
 אתה לא שואל יותר שאלות לגבי הצרכים של לקוח'Спец_по_завершению'אם אתה רואה שיש הודעות של
'''

#@title 5. 'Спец_по_завершению'
name_spez6 = 'Спец_по_завершению'
model_spez6 = MODEL
temperature_spez6 = 0
verbose_spez6 = 0
system_prompt_spez6 = '''
 אתה מסיים את השיחה ונפרד מהלקוח.
 המשימה שלך: לענות כאשר השיחה הגיעה לסיומה - כאשר הלקוח מפסיק לשאול שאלות חדשות
 או לא כתב בהודעתו דבר מלבד ביטויים כמו "תודה", "ברור" או ביטויי פרידה. אתה יודע שביטויי פרידה הם ביטויים או משפטים המשמשים לסיום שיחה או דיאלוג.
ביטויי פרידה משמשים לעיתים קרובות להעברת הבנה,
 הסכמה, הכרת תודה או רצון להיפרד (למשל להתראות, שלום, נתראה בקרוב, כל טוב, להתראות בקרוב). לביטויי פרידה יש טון ורמת פורמליות שונים, והבחירה בביטוי מסוים תלויה בהקשר.
 אתה תמיד עוקב בקפידה אחר הכרונולוגיה של ההודעות הקודמות בשיחה, כדי להפוך את תשובתך להיגיונית לכרונולוגיה זו.
  אסור לך לשאול את הלקוח שאלות. אסור לך לענות על בקשות הלקוח. המשימה שלך היא לכתוב ללקוח הודעה המסיימת את השיחה, לאשר את ההסכמות לפגישה (אם היו) ולהיפרד מהלקוח.
אם הלקוח אומר שהוא רוצה להזמין תענה שאתה מעביר אותו לתהליך הזמנה. אחרי זה אתה תשאל אותו אם הוא מעוניין להזמין עוד משהו.
ואם הוא לא מעוניין אתה נפרד ממנו לשלום.

'''
instructions_spez6 = '''נתח את שאלת הלקוח ואת הכרונולוגיה של ההודעות הקודמות בשיחה: אם הלקוח שאל שאלה שאינה קשורה למוצרי המאפייה, ענה שאין לך מידע לענות, ובשאר המקרים כתוב ללקוח הודעה המסיימת את השיח.
אם הלקוח מבקש לעשות הזמנה תעביר אותו לתהליך הזמנה, ותשאל אם הוא צריך עוד משהו. אם הוא עונה בשלילה-תיפרד ממנו לשלום ואל תשאל יותר שאלות

'''

"""### Функция"""

def spez_user_question(name, system, instructions, question, summary_history, summary_exact, base_topicphrase, search_index, temp=0, verbose=0, k=num_chunks, model=MODEL):
    if name in ["Zoom_Пуш", "Спец_по_завершению"]:
      docs_content = ''
    else:
      knowledge_base = search_index.similarity_search(base_topicphrase, k=k)
      docs_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\n==================\n' + doc.page_content + '\n' for doc in knowledge_base]))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f'''{instructions}

         Вопрос клиента:{question}

         Хронология предыдущих сообщений диалога: {summary_history}

         Точное саммари: {summary_exact}

         База Знаний: {docs_content}'''}
    ]
    if verbose: print('\n==================\n')
    if verbose: print(f'{bcolors.BGCYAN}Вопрос клиента:{bcolors.ENDC}', question)
    if verbose: print('Саммари диалога:\n==================\n',
                         summary_history)
    if verbose: print(f'{bcolors.BGYELLOW}Саммари точное:{bcolors.ENDC}\n==================\n',
                         summary_exact)
    if verbose: print(f'{bcolors.BGGREEN}База знаний:{bcolors.ENDC}\n==================\n', docs_content)

    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    answer = completion.choices[0].message.content
    try:
      answer = answer.split(': ')[1]+ ' '
    except:
      answer = answer
    answer = answer.lstrip('#3')
    if verbose: print(f'\n==================')
    if verbose: print(f'{completion.usage.total_tokens} total tokens used (question-answer).')
    if verbose: print('\n==================\n')
    if verbose: print(f'{bcolors.RED}{bcolors.BOLD}Ответ {name}:{bcolors.ENDC}\n',
                      f'{bcolors.RED}{answer}{bcolors.ENDC}')
    return completion

"""### Конфигурация"""

spez_config ={
    'Обработчик_возражений': {
        'name':name_spez2,
        'system':system_prompt_spez2,
        'instructions':instructions_spez2,
        'k': num_chunks,
        'temp':temperature_spez2,
        'verbose': verbose_spez2,
        'model':model_spez2,
        },
    'Спец_по_презентациям': {
        'name':name_spez3,
        'system':system_prompt_spez3,
        'instructions':instructions_spez3,
        'k': num_chunks,
        'temp':temperature_spez3,
        'verbose': verbose_spez3,
        'model':model_spez3,
        },
    'Zoom_Пуш': {
        'name':name_spez4,
        'system':system_prompt_spez4,
        'instructions':instructions_spez4,
        'k': num_chunks,
        'temp':temperature_spez4,
        'verbose': verbose_spez4,
        'model':model_spez4,
        },
    'Спец_по_выявлению_потребностей': {
        'name':name_spez5,
        'system':system_prompt_spez5,
        'instructions':instructions_spez5,
        'k': num_chunks,
        'temp':temperature_spez5,
        'verbose': verbose_spez5,
        'model':model_spez5,
        },
    'Спец_по_завершению': {
        'name':name_spez6,
        'system':system_prompt_spez6,
        'instructions':instructions_spez6,
        'k': num_chunks,
        'temp':temperature_spez6,
        'verbose': verbose_spez6,
        'model':model_spez6,
        }

}

"""## user_question_router - Диспетчер-маршрутизатор

модель определяет по контексту, Хронологии предыдущих сообщений диалога и точному саммари каких узких специалистов нужно привлечь для подготовки материалов для проактивного ответа Старшего менеджера
"""

#@title Параметры
name_router = 'Диспетчер-маршрутизатор'
model_router = MODEL
temperature_router = 0
verbose_router = 0

# если уже выявлено 4 потребности, то больше потребности вывлять не нужно.
# это можно сделать в промпте, но он перегружен. Поэтому ограничим вызов спеца по выявлению программно:
system_prompt_router ='''
אתה מבצע את המשימה שלך בצורה מושלמת:
 אתה קובע לאילו מומחים יש לפנות כדי ליצור תשובה נכונה ללקוח. אתה יודע שניתן לפנות רק למומחים מהרשימה:

 '''
instructions_router ='''
בואו נפעל בשלבים:
שלב 1: נתחו את שאלת הלקוח ואת הכרונולוגיה של ההודעות הקודמות בשיחה כדי להיות בהקשר;
 שלב 2: נתחו את הסיכום המדויק - הוא מכיל בקצרה את הצרכים שכבר זוהו, ההתנגדויות שטופלו והתעריפים שהוצגו;
  שלב 3: בהתבסס על הניתוח של שלב 1 ושלב 2,
   כתבו רשימת מומחים למתן תשובה ללקוח. ענו בבקשה במדויק, ואל תמציאו שום דבר בעצמכם. רשימת המומחים
  python list יכולה להיות ריקה [] אם אין צורך, או מומחה אחד, או כמה, או כולם. סדר הדיווח: כתבו רק את רשימת המומחים משלב 3 בפורמט


'''

#@title Функция
def user_question_router(name, system, instructions, question, summary_history, summary_exact, temp=0, verbose=0, model=MODEL, needs_lst=[]):
    if verbose: print('\n==================\n')
    if verbose: print( question)
    if verbose: print('Саммари диалога:\n==================\n',
                         summary_history)
    if verbose: print(f'{bcolors.BGYELLOW}Саммари точное:{bcolors.ENDC}\n==================\n',
                         summary_exact)


    if needs_lst and len(needs_lst) > 5:
      system +=''' ["Обработчик_возражений", "Спец_по_презентациям", "Zoom_Пуш", “Спец_по_завершению”].
      אתה יודע על מה כל מומחה אחראי:
     מטפל בהתנגדויות: מומחה זה משתתף בתשובה ללקוח אם:
        הלקוח הביע התנגדות או ספק;
          הלקוח אינו מרוצה או לא מרוצה מהמוצר;
           מומחה למצגות: מומחה זה משתתף בתשובה
       ללקוח אם הלקוח הביע עניין בקניית מוצרי מאפייה , אם הוא כבר הציג זאת בהודעות
       הקודמות של הדיאלוג, אז אסור להציג שוב;
       מומחה זה משתתף  Zoom_Push:

      בתשובה ללקוח כאשר:  הלקוח מוכן לרכוש את המוצר וצריך לקדם את הלקוח אל השלמת העסקה הרכישה;  ב-
       מספק את פרטי הקשר שלו לשליחת הזמנה לפגישה ב- מומחה לסיום: מומחה זה משתתף בתשובה ללקוח בסוף הדיאלוג, תפקידו לענות כאשר המשתמש
       נותן להבין שהוא מסיים את הדיאלוג ואינו מתכוון לשאול עוד שאלות, למשל: "תודה", "הכל ברור", "בסדר", "טוב" וביטויים מאשרים אחרים המסיימים לוגית את התקשורת.

'''
    else:
      system +=''' ["Спец_по_выявлению_потребностей", "Обработчик_возражений", "Спец_по_презентациям", "Zoom_Пуш",  “Спец_по_завершению”].
       הנה תיאור המומחים:

        מומחה לזיהוי צרכים: מומחה זה תמיד משתתף בתשובה;
        אלה אם כן הלקוח כבר ביצע רכישה
         מטפל בהתנגדויות: מומחה זה משתתף בתשובה ללקוח אם:
          הלקוח הביע התנגדות או ספק;
         הלקוח אינו מרוצה או לא מרוצה מהמוצר;
          מומחה למצגות: מומחה זה משתתף בתשובה ללקוח אם הלקוח הביע עניין באחד המוצרים של המאפיה וצריך לספר לו עליו
          אם הוא כבר הציג זאת בהודעות הקודמות של הדיאלוג,
         אז אסור להציג שוב;
         מומחה זה משתתף בתשובה ללקוח כאשר:Zoom_Push:
        הלקוח אומר שהמוצר מוצא חן בעיניו ואז צריך להוביל את הלקוח בעדינות לעיסקה;
       הלקוח מביע נכונות לרכוש מוצר או שירות ה;

       מומחה לסיום:
       מומחה זה משתתף בתשובה ללקוח בסוף הדיאלוג, תפקידו לענות כאשר המשתמש נותן להבין
        שהוא מסיים את הדיאלוג ואינו מתכוון לשאול עוד שאלות, למשל: "תודה", "הכל ברור", "בסדר", "טוב" וביטויים מאשרים אחרים המסיימים לוגית את התקשורת.אם המומחה לסיום פעל אז צריך רק לשאול את הלקוח אם יש עוד משהו שהוא רוצה להזמין וחוץ מזה אסור לשאול יותר את הלקוח שאלות'''

      system += '''  המשימה שלך: לקבוע על פי הודעת הלקוח, על סמך הידע שלך,
       הסיכום המדויק והכרונולוגיה של ההודעות הקודמות בשיחה, אילו מומחים מהרשימה יש לבחור, כדי שהם ישתתפו בתשובה ללקוח. אתה תמיד עוקב בקפידה אחר הדרישות לסדר הדיווח.
        '''
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f'''{instructions}
                                      \n\nВопрос клиента:{question}
                                      \n\nХронология предыдущих сообщений диалога: {summary_history}
                                      \n\nСаммари точное: {summary_exact}
                                      \n\nОтвет: '''}
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )
    answer = completion.choices[0].message.content
    if verbose: print(f'\n==================')
    if verbose: print(f'{completion.usage.total_tokens} total tokens used (question-answer).')
    if verbose: print('\n==================\n')
    if verbose: print(f'{bcolors.RED}{bcolors.BOLD}Ответ {name}:{bcolors.ENDC}\n',
                      f'{bcolors.RED}{answer}{bcolors.ENDC}')
    if verbose: print(f'{bcolors.RED}{answer}{bcolors.ENDC}')
    #print(f'{bcolors.RED}{answer}{bcolors.ENDC}')
    return completion

"""## senior_answer - генерация старшим менеджером проактивного ответа на основании данных узких специалистов"""

#@title Параметры
name_senior = 'Старший менеджер по продажам'
model_senior = MODEL
temperature_senior = 0
verbose_senior = 0
num_fragments_base_senior = num_chunks
system_prompt_senior = f'''
{knowledge_db_txt}  לפני שאתה מתחיל לענות אתה בודק אם המוצר קיים ב
אבל אם יש אתה מציע אותו
אם המוצר לא קיים אתה מתנצל ולא מציג תשובות של 'Спец_по выявлению_потребностей'
 מי אתה: אתה אדם, גבר, שמך אילן. אתה מנהל המכירות הבכיר הטוב ביותר. אתה עובד ב"מאפיית שלמה", שמתמחה במכירת מאפים. אתה מבין לא רע במוצרים
 ובתנאי השירות של המאפייה.
 בעת מתן תשובה אתה בשום
 אופן לא ממציא משהו משלך.אתה תמיד עוקב אחר ההיגיון של הכרונולוגיה של ההודעות הקודמות בשיחה כדי שתשובתך
   אתה כולל בתשובה שלך את כל המידע שהוא סיפק, 'Спец_по_презентациям' ללקוח תהיה הגיונית ועקבית, ובראש
   ובראשונה אתה מתמקד בתשובות של מומחים:'''

instructions_senior = f'''{knowledge_db_txt}  לפני שאתה מתחיל לענות אתה בודק אם המוצר קיים ב

אבל אם יש אתה מציע אותו
אם המוצר לא קיים אתה מתנצל ואומר שאין כרגע את המוצר הזה
בואו נחשוב בצורה עקבית. עליכם לפתור את המשימות הבאות:  1: נתחו את תשובות המומחים - זו המידע שעליו אתם מסתמכים בתשובה;
 משימה 2: נתחו את הכרונולוגיה של ההודעות הקודמות בשיחה כדי לכתוב את תשובתכם בצורה עוקבת והגיונית לכרונולוגיה זו;
משימה 3: נתחו את הסיכום המדויק כדי להבין אילו צרכים,
 ואילו התנגדויות כבר טופלו במהלך הדיאלוג ולא לחזור על עצמכם; משימה 4: בהתבסס על הניתוח שלכם במשימות 1-3 ובכפוף להוראות,
 כתבו תשובה הגיונית, קצרה ועקבית ללקוח, תוך חיקוי תשובת אדם אמיתי בכפוף לסדר הדיווח: בתשובתכם אל תספרו ללקוח על מטרותיכם;
 תמיד כתבו טקסט בצורה לא פורמלית, אל תקראו ללקוח "לקוח", אם יש מידע על שם הלקוח בכרונולוגיה של ההודעות הקודמות בשיחה,
הוסיפו את שם הלקוח לתשובתכם. בעת כתיבת תשובתכם אסור לכם לחזור על טענות שכבר הועלו בכרונולוגיה של ההודעות הקודמות בשיחה.
בשום אופן אתה לא ממציא דברים. אם אתה נשאל על משהו שאין בתוך בסיס הידע תענה שאתה לא יודע את התשובה
'''

#@title Функция
def senior_answer(name, system, instructions, question, output_spez,
                                   summary_history, base_topicphrase, search_index, summary_exact, temp=0,
                                   verbose=0, k=num_chunks, model=MODEL, spez_list = []):
    # дозаполним system по набору спецов
    if 'Спец_по_завершению'  in spez_list:
        system += "Спец_по_завершению."
    else:
        system += '''Обработчик_возражений, Спец_по_презентациям, Zoom_пуш, Спец_по_выявлению_потребностей.
 מטרת השיחה שלך: במהלך כל השיחה לזהות את הצרכים של הלקוח, לסגור את כל ההתנגדויות של הלקוח
ולבסוף לשכנע אותו לקנות את המוצרים של מאפיה. אתה תמיד פועל לפי ההוראות ובסדר הדיווח בצורה קפדנית.

 הוראות כיצד לענות על שאלות הלקוח:
 בעת ניסוח תשובתך, תמיד פעל לפי הלוגיקה של כרונולוגיית ההודעות הקודמות בשיחה והסתמך על תשובות המומחים בתחום.
 הצג את המוצרים רק אם הלקוח ביקש לדעת על מוצר כלשהו או אם הוא  מספר צורך כלשהו. הצג את המצגת בהסתמך על תשובת המומחה_במצגות;
 אם יש לך תשובת "מנהל התנגדויות", סגור את ההתנגדויות בהסתמך על תשובת "מנהל התנגדויות";
  אתה יודע שחשוב לסגור את כל ההתנגדויות של הלקוח;

בתשובתך אסור לך לומר שאתה מגלה את הצרכים והמטרות של הלקוח;

  אסור לך לשוחח על נושאים שאינם קשורים למוצרים או שירותים של המאפיה.

 הוראות כיצד להתייחס לשאלות על נושאים שאינם קשורים: אם בתשובות המומחים כתוב ששאלה אינה קשורה למאפיה או מוצריה ל-
 זה אומר שעליך לסרב בעדינות לענות על שאלות שאינן קשורות ולברר אם יש ללקוח שאלות נוספות ב-

'''

    if 'Zoom_Пуш'  in spez_list:
        system += '''
אתה אמור להבין מתוך השיחה באיזה שלב לקראת הרכישה הלקוח נמצא ולפי זה לקדם אותו בעדינות אל העיסקה'''
    system += '''
אתה תמיד פועל לפי הסדר המדויק של הדיווח'''

    # дозаполним instructions по набору спецов
    if 'Спец_по_завершению'  not in spez_list:
      if 'Спец_по_выявлению_потребностей' in spez_list:
        instructions += '''
 Спец_по_выявлению_потребностей",משימה: בהתבסס על הניתוח שלך, בחר רק שאלה אחת מתוך תשובת
  שאין אותה בהיסטוריית ההודעות הקודמות של השיחה,
  והיא מתאימה ביותר ללוגיקה של ההיסטוריה הקודמת של השיחה.
'''
      else:
        instructions += '''
 המשימה: בהתבסס על הניתוח שלך, שאל שאלה שתסייע להמשך השיחה, תוך שמירה על הלוגיקה של היסטוריית ההודעות הקודמות.'''
      instructions += '''  אל תסבירו את הבחירה שלכם ואל תעירו דבר, אל תסבירו מאילו תשובות של מומחים אתם יוצרים את התשובה שלכם.
       סדר הדיווח: בתשובתכם צריכה להיות רק תשובה ללקוח (משימה 4) + רק שאלה ללקוח (משימה 5) (ללא הסברים והערות).

'''
    else:
      instructions += '''
ה  סדר הדיווח: בתשובתכם חייבת להיות רק התשובה ללקוח"Спец_по_завершению".בתשובתכם יש לכלול רק את התשובת
'''
    knowledge_base = search_index.similarity_search(base_topicphrase, k)
    docs_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\n=====\n' + doc.page_content + '\n' for doc in knowledge_base]))
    output_spez_content = "\n=====\n".join(output_spez)
    if verbose:
      print(f'{bcolors.BGGREEN}Ответы узких специалистов:{bcolors.ENDC}{output_spez_content}\n')
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f'''{instructions}
                                    \nВопрос клиента: {question}
                                    \nХронология предыдущих сообщений диалога: {summary_history}
                                    \nСаммари точное: {summary_exact}
                                    \nОтветы узких специалистов: {output_spez_content}
                                      \nДополнительная информация: {docs_content}
                                    \nОтвет: '''}
    ]
#
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )

    if verbose:
      print(f'{bcolors.OKBLUE}{bcolors.BOLD}Ответ {name}:{bcolors.ENDC}\n',
            f'{bcolors.OKBLUE}{completion.choices[0].message.content}{bcolors.ENDC}')

    return completion

"""## Стилист"""

#@title Параметры
name_stilist = 'Стилист'
model_stilist = MODEL
temperature_stilist = 0
verbose_stilist = 0
system_prompt_stilist = '''
אתה אדם צעיר, גבר, ואתה מתנהל בשיחה בצורה מעניינת עבור השותף לשיחה.
אתה אוהב את "מאפיית שלמה". יש לך דיבור מאוד רהוט, ואתה נמנע מלהשתמש במילים מאותה משפחת מילים באותו משפט, ונמנע משימוש במילים חוזרות במשפטים סמוכים.
אתה יודע שסטיליזציה היא לעצב את הטקסט בסגנון שלך תוך שמירה על משמעותו ומסריו.
 הסגנון שלך אנושי, תמציתי ומשכנע, ואינך משתמש בקלישאות וביטויים תבניתיים, כמו "הכבוד הרב", "לקוח" וכו'.
  אתה תמיד שומר על הכללים הללו. אם בטקסט המקורי יש הצגה של מוצרים אן שירותים אתה שומר אותם. תשמור גם את שמו של הלקוח אם הוא אמר אותו. המשימה שלך: לעצב את הטקסט המקורי כך שיתקבל חיקוי של שיחה אמיתית עם אדם.
'''
instructions_stilist = '''

  אם בטקסט המקורי יש הצגה של מוצרים אן שירותים אתה שומר אותם.
  אבל מוציא כל מיני סימנים כמו למשל * ודברים דומים ומציג את המידע בצורה טבעית וקלה לקריאה
בואו נחשוב שלב אחרי שלב:

שלב 1: ניתוח הטקסט המקורי;
שלב 2: לעצב את הטקסט המקורי בסגנונכם, תוך שמירה על כל  הכללים ב-100%:
 אם בטקסט המקורי יש ברכות או איחולים להצלחה, הוציאו אותם מהתשובה שלכם;
 אם בטקסט המקורי יש שאלות סגורות, שנו אותן לשאלות פתוחות;
 אסור לכם לקרוא ללקוח לקוח
סדר הדיווח: בתשובתכם כלול רק הטקסט המסוגנן.

'''

#@title Функция
def stilizator_answer(name, system, instructions, answers_content, temp=0, verbose=0, model=MODEL):

    if verbose: print('==================')
    if verbose: print(f'{bcolors.BGCYAN}Текст для стилизации:{bcolors.ENDC}\n{answers_content}')
    user_assist = f'''{instructions}\n\nטקסט מקור: אילנה, אני שמח שהתעניינת במאפייה שלנו.
     אנו מציעים מאפים טריים כל יום וגאים במתכונים שעוברים מדור לדור. אילו סוגי מאפים תרצי לנסות?

תשובה:
'''

    user_assist2 = f'''{instructions}\n\nטקסט מקור: במאפייה שלנו תמצאו את המבחר הרחב ביותר של מאפים, כולל קרואסונים קלאסיים,
    חלות ועוגות ייחודיות. אנו משתמשים רק במרכיבים טבעיים ומציעים גם משלוחים. מה תרצו לטעום?

תשובה:
'''


    messages = [

        {"role": "system", "content": system},
        {"role": "user", "content": user_assist},
       {"role": "assistant", "content": '''במאפייה שלנו יש מבחר רחב, כולל חלות, קרואסונים ועוגות,
        הכל עם מרכיבים טבעיים בלבד. אנו גם מציעים משלוחים. מה מהמאפים שלנו היית רוצה לנסות?'''},

        {"role": "user", "content": user_assist2},
        {"role": "assistant", "content": '''במאפייה שלנו יש מבחר רחב, כולל חלות, קרואסונים ועוגות,
         הכל עם מרכיבים טבעיים בלבד. אנו גם מציעים משלוחים. מה מהמאפים שלנו היית רוצה לנסות?'''},
        {"role": "user", "content": f'''{instructions}\n\n טקסט מקורי  : {answers_content}\n\n תשובה : '''}
    ]

    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )

    if verbose: print('\n==================')
    if verbose: print(f'{completion.usage.total_tokens} total tokens used (question-answer).')
    if verbose: print('==================')
    if verbose:
      print(f'{bcolors.BGMAGENTA}Ответ {name}:{bcolors.ENDC}\n',
            f'{bcolors.HEADER}{completion.choices[0].message.content}{bcolors.ENDC}')

    return completion

"""## Очистка от приветствия"""

#@title Параметры:
system_prompt_stilist1 = '''
אתה עורך טקסטים מצוין וטוב יותר מכולם במציאת ברכת פתיחה בטקסט.
ברכת פתיחה היא ביטוי של ברכה או הודעת פתיחה שנשלחת או נאמרת בתחילת שיחה עם מישהו.
ברכת פתיחה יכולה להיות רשמית או לא רשמית, תלויה בתרבות ובהקשר.
היא משמשת להפגנת נימוס, ידידות ורצון ליצור קשר עם האדם שמולך.
ברכות פתיחה יכולות להיות שונות בשפות ובתרבויות שונות, החל מ"שלום" או "ברוך הבא" הפשוטות ועד ביטויים מסורתיים או רשמיים יותר.
המטרה שלך היא לעבד את הטקסט המקורי באופן הבא:
לנתח את הטקסט המקורי, ואם יש בו ברכת פתיחה, למחוק אותה.
בתשובתך כלול את הטקסט המעובד ללא ברכת הפתיחה.
'''
instructions_prompt_stilist1 = ''

#@title Функция
def del_hello(name, system, instructions, answers_content, temp=0, verbose=0, model=MODEL):

    if verbose: print('==================')
    if verbose: print(f'{bcolors.BGCYAN}Текст для стилизации:{bcolors.ENDC}\n{answers_content}')
    user_assist = '''\n\nשלום אילנה, אני אשמח לספר לך על המוצרים שלנו. על מה תרצי לשמוע קודם?\n\nתשובה:'''
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_assist},
        {"role": "assistant", "content":'''\n\ אילנה, אני אשמח לספר לך על המוצרים שלנו. על מה תרצי לשמוע קודם?'''},
        {"role": "user", "content": f'''\n\nИсходный текст: {answers_content}\n\nОтвет: '''}
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp
    )

    if verbose: print('\n==================')
    if verbose: print(f'{completion.usage.total_tokens} total tokens used (question-answer).')
    if verbose: print('==================')
    if verbose:
      print(f'{bcolors.BGMAGENTA}Ответ {name}:{bcolors.ENDC}\n',
            f'{bcolors.HEADER}{completion.choices[0].message.content}{bcolors.ENDC}')

    return completion

"""# Функции Python"""

#@title Общие функции
# Функция загружает plane text из ГуглДока по URL-ссылке (url).
# ГуглДок должен быть открыт для чтения всем, у кого есть ссылка
def load_googledoc_by_url(url: str) -> str:
        match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
        if match_ is None:
            raise ValueError('Invalid Google Docs URL')
        doc_id = match_.group(1)
        response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
        response.raise_for_status()
        return response.text

# Функция подсчета токенов для модели эмбеддингов
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# функция подсчета токенов в сообщении модели
def num_tokens_from_messages(messages, model):
      """Returns the number of tokens used by a list of messages."""
      try:
          encoding = tiktoken.encoding_for_model(model)
      except KeyError:
          encoding = tiktoken.get_encoding("cl100k_base")
      if model in ["gpt-4-1106-preview", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106"]:  # note: future models may deviate from this
          num_tokens = 0
          for message in messages:
              num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
              for key, value in message.items():
                  num_tokens += len(encoding.encode(value))
                  if key == "name":  # if there's a name, the role is omitted
                      num_tokens += -1  # role is always required and always 1 token
          num_tokens += 2  # every reply is primed with <im_start>assistant
          return num_tokens
      else:
          raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
          See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


# Функция разделения текста на чанки заданной длины (в токенах)
def split_text(text, max_count, chunk_overlap):

    # Функция для подсчета количества токенов во фрагменте для сплиттера RecursiveCharacterTextSplitter
    def num_tokens(fragment):
        return num_tokens_from_string(fragment, "cl100k_base")

    num_levels = 3  # Число уровней заголовков, которые будем разделять сплиттером MarkdownHeaderTextSplitter

    headers_to_split_on = [
    (f"{'#' * i}", f"H{i}") for i in range(1, num_levels + 1)
    ]
    # сначала разделим с помощью MarkdownHeaderTextSplitter
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    fragments = markdown_splitter.split_text(text)


    # дальше будем делить делить каждый полученный чанк вторым сплиттером RecursiveCharacterTextSplitter
    # 1. для того, чтобы быть уверенным в размере полученного чанка
    # 2. база знаний размечена так, что заголовки не повторяются в тексте разделов, мы это исправим принудительно:

    splitter = RecursiveCharacterTextSplitter(chunk_size=max_count, chunk_overlap=0, length_function=num_tokens)

    source_chunks = []

    # Обработаем каждый фрагмент текста, полученный после MarkdownHeaderTextSplitter
    for fragment in fragments:
    # MarkdownHeaderTextSplitter сохранил иерархию заголовков в Метаданных - вытащим ее
      level = 0
      headers = ['','','','']
      for j in range(1, num_levels + 1):
        header_key = f'H{j}'
        if header_key in fragment.metadata: level, headers[j-1] = j, fragment.metadata[header_key]
      header_string = ' '.join([f"{'#' * i} {header}" for i, header in enumerate(headers[:level], start=1)])

    # каждый фрагмент будем разбивать на чанки с помощью RecursiveCharacterTextSplitter
    # допишем иерархию заголовков в конец чанка
    # унаследуем метаданные от первого сплиттера
    # добавим в метаданные размер чанка в токенах
      for i,chunk in enumerate(splitter.split_text(fragment.page_content)):
        mdata = fragment.metadata.copy()
        add_hierarchy = f'{header_string}: уровень {level} пункт {i+1}'
        new_chunk = ' '.join([chunk, f'\nРаздел: {add_hierarchy}'])
        mdata["len"] = num_tokens(new_chunk)
        doc = Document(page_content=new_chunk, metadata=mdata)
        source_chunks.append(doc)

    return source_chunks

# функция для очистки списка строк от повторений (для формирования более корректного точного саммари)
def list_cleaner(list_to_clean):
  filtered_list = [value.replace('"', '').strip() for value in list_to_clean if value.replace('"', '').strip() != '']
  text = ', '.join(filtered_list).replace('\n', ',').replace('-', ' ')
  phrases = text.split(',')
  return list(set(map(str.strip, text.split(','))))

# Запуск модели  суфлера при первом обращении клиента
def sufler(history_user):
  hello_completion = get_hello(MODEL,history_user[0],0.4,0)
  hello = hello_completion.choices[0].message.content
  try:
      hello_word = str(hello).split(': ')[1]+ ' '
  except:
      hello_word = str(hello)

  hello_word = hello_word.capitalize().rstrip(string.punctuation) + ', '

  if 'None' in hello_word: hello_word='שלום'  # если пользователь не поздоровался, то мы поздороваемся нейтрально-формально

  return hello_word

# Функция удаления перехода на новую строку
def remove_newlines(text):
    cleaned_string = text.replace('\n', ' ')
    return cleaned_string

#фунция добавления переходов на новую строку для удобства чтения
# здесь в параметр text подается текст без переносов строки
# функция добавит переходы на новую строку при достижении max_len символов
def insert_newlines(text: str, max_len: int = 160) -> str:
      words = text.split()
      lines = []
      current_line = ""
      for word in words:
          if len(current_line + " " + word) > max_len:
              lines.append(current_line)
              current_line = ""
          current_line += f' {word}'
      lines.append(current_line)
      return "\n".join(lines)

#функция добавления переходов на новую строку для удобства чтения
# здесь в параметр text подается текст, содержащий переносы строки
# функция добавит переходы на новую строку при достижении max_len символов
def insert_newlines_n(text: str, max_len: int = 150) -> str:
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        words = line.split()
        current_line = ""
        for word in words:
            if len(current_line + " " + word) > max_len:
                new_lines.append(current_line)
                current_line = ""
            current_line += f' {word}'
        new_lines.append(current_line)
    return "\n".join(new_lines)

#@title Ансамбль моделей для формирования ответа нейро-продажника

def get_seller_answer(history_user, history_manager, history_chat):
    output_router_list = []

    global needs_extractor
    global benefits_extractor
    global objection_detector
    global resolved_objection_detector
    global tariff_detector

    global summarized_dialog


    # 1. Выделим потребности из последнего сообщения клиента  и добавим в историю потребностей
    output_ne = extract_entity_from_statement(
                      name=name_needs_extractor,
                      system=system_prompt_needs_extractor,
                      instructions=instructions_needs_extractor,
                      question=history_user[-1],
                      history='',
                      temp=temperature_needs_extractor,
                      verbose=verbose_needs_extractor,
                      model=model_needs_extractor).choices[0].message.content
    try:
      output_needs_extractor = str(output_ne).split(':')[1]+ ''
    except:
      output_needs_extractor = str(output_ne)
    needs_extractor.append(output_needs_extractor)
    needs_extractor = list_cleaner(needs_extractor)

    # 2. Выделим названные менеджером преимущества, добавим в историю озвученных преимуществ
    output_be = extract_entity_from_statement(
                      name=name_benefits_extractor,
                      system=system_prompt_benefits_extractor,
                      instructions=instructions_benefits_extractor,
                      question='',
                      history=history_manager,
                      temp=temperature_benefits_extractor,
                      verbose=verbose_benefits_extractor,
                      model=model_benefits_extractor).choices[0].message.content
    try:
      output_benefits_extractor = str(output_be).split(':')[1]+ ''
    except:
      output_benefits_extractor = str(output_be)
    benefits_extractor.append(output_benefits_extractor)
    benefits_extractor = list_cleaner(benefits_extractor)

    #3. Выделим наличие возражений, добавим их в историю возражений
    output_obj = extract_entity_from_statement(
                      name=name_objection_detector,
                      system=system_prompt_objection_detector,
                      instructions=instructions_objection_detector,
                      question=history_user[-1],
                      history='',
                      temp=temperature_objection_detector,
                      verbose=verbose_objection_detector,
                      model=model_objection_detector).choices[0].message.content
    try:
        output_objection_detector = str(output_obj).split(':')[1]+ ''
    except:
        output_objection_detector = str(output_obj)
    objection_detector.append(output_objection_detector)
    objection_detector = list_cleaner(objection_detector)

    #4. Выделим отработанные менеджером возражения клиента, добавим их в историю отработанных возражений
    output_res = extract_entity_from_statement(
                      name=name_resolved_objection_detector,
                      system=system_prompt_resolved_objection_detector,
                      instructions=instructions_resolved_objection_detector,
                      question='',
                      history=history_manager,
                      temp=temperature_resolved_objection_detector,
                      verbose=verbose_resolved_objection_detector,
                      model=model_resolved_objection_detector).choices[0].message.content
    try:
        output_resolved_objection_detector = str(output_res).split(':')[1]+ ''
    except:
        output_resolved_objection_detector = str(output_res)
    resolved_objection_detector.append(output_resolved_objection_detector)
    resolved_objection_detector = list_cleaner(resolved_objection_detector)

    #5. Выделим названные менеджером тарифы и цены, добавим их в историю названных тарифов и цен
    output_tar = extract_entity_from_statement(
                      name=name_tariff_detector,
                      system=system_prompt_tariff_detector,
                      instructions=instructions_tariff_detector,
                      question='',
                      history=history_manager,
                      temp=temperature_tariff_detector,
                      verbose=verbose_tariff_detector,
                      model=model_tariff_detector).choices[0].message.content
    try:
        output_tariff_detector = str(output_tar).split(':')[1]+ ''
    except:
        output_tariff_detector = str(output_tar)
    tariff_detector.append(output_tariff_detector)
    tariff_detector = list_cleaner(tariff_detector)

    #6. Выделим ключи из последних сообщений клиента и менеджера (предыдущий вопрос+ответ)
    k = 2 if len(history_user)>1 else 1
    if history_manager and len(history_manager)>0:
      manager_list = history_manager[-1:]
    else:
      manager_list = []

    topicphrase_completion = get_topicphrase_questions(name=name_todo_base,
                                _user=history_user[-k:],
                                _manager = manager_list,
                                system=system_topicphrase_extractor,
                                instruction=instructions_topicphrase_extractor,
                                temp=temperature_todo_base,
                                verbose=verbose_base,
                                model=model_todo_base)
    topicphrase_answ = topicphrase_completion.choices[0].message.content

    # именно general_topic_phrase будем подавать в Langchain для similary search, преобразуем в строку-перечисление через запятую + добавляем текущий вопрос клиента
    general_topic_phrase = str(history_user[-1]+', ' + topicphrase_answ).replace('[','') .replace(']','').replace("'",'').replace("'",'')

    #7. Суммаризируем хронологию предыдущих сообщений диалога
    summarized_comp = summarize_dialog(summarized_dialog, history_chat, temp=0, verbose=verbose_router, model=MODEL)
    summarized_dialog = summarized_comp.choices[0].message.content

    #8. Создаем точное саммари с ключевыми моментами диалога
    tochnoe_summary = f'''
# 1. Выявлены Потребности: {', '.join(needs_extractor) if needs_extractor else 'потребностей не обнаружено'}
# 2. Расказанные Преимущества: {', '.join(benefits_extractor) if benefits_extractor else 'преимущества не были рассказаны'}
# 3. Возражения клиента: {', '.join(objection_detector) if objection_detector else 'возражений не обнаружено'}
# 4. Возражения клиента отработаны: {', '.join(resolved_objection_detector) if resolved_objection_detector else 'отработки не обнаружено'}
# 5. Конкретика - оговоренная конкретика - курсы, цены: {', '.join(tariff_detector) if tariff_detector else 'не обнаружено'}
'''
    #9. Запускаем Диспетчера
    output_router = user_question_router(name= name_router,
                                        system=system_prompt_router,
                                        instructions=instructions_router,
                                        question=history_user[-1],
                                        summary_history=summarized_dialog,
                                        summary_exact=tochnoe_summary,
                                        temp=temperature_router,
                                        verbose=verbose_router,
                                        model=model_router,
                                        needs_lst=needs_extractor).choices[0].message.content

    #10. По списку спецов из ответа Диспетчера запускаем спецов:
    output_spez = []
    try:
        output_router_fixed = (str(output_router).split(':')[1]+ '').replace("‘", '"').replace("'", '"')
    except:
        output_router_fixed = str(output_router).replace("‘", '"').replace("'", '"')

    try:
      output_router_list = json.loads(output_router_fixed)
    except:
      output_router_list = ['Zoom_Пуш', 'Спец_по_презентациям']
    #print(f'{bcolors.RED}{output_router_list}{bcolors.ENDC}')
    try:
      for key_param in output_router_list:
          param = spez_config[key_param] | {'question': history_user[-1],
                                      'summary_history': summarized_dialog,
                                      'summary_exact': tochnoe_summary,
                                      'base_topicphrase': general_topic_phrase,
                                      'search_index': vectordb}
          spez_answer = spez_user_question(**param).choices[0].message.content
          try:
            answer = spez_answer.split(': ')[1]+ ' '
          except:
            answer = spez_answer
          answer = answer.lstrip('#3')

          output_spez.append(f'{param["name"]}: {answer}')
      verbose=False
      if verbose: print(f"\n{bcolors.BGMAGENTA}Ответы спецов:{bcolors.ENDC}\n", '\n\n=========\n'.join(output_spez))
    except:
      if verbose: print(f'{bcolors.BGYELLOW}Ответ диспетчера либо не вызывает спецов либо имеет неверный формат:{bcolors.ENDC} {output_router}')


    #11. На основании предлоажения узких спецов запускаем страшего менеджера для подготовки комплексного ответа:
    output_senior = senior_answer(
                          name=name_senior,
                          system=system_prompt_senior,
                          instructions=instructions_senior,
                          question=history_user[-1],
                          output_spez=output_spez,
                          summary_history=summarized_dialog,
                          base_topicphrase=general_topic_phrase,
                          search_index=vectordb,
                          summary_exact=tochnoe_summary,
                          temp=temperature_senior,
                          verbose=verbose_senior,
                          k=num_fragments_base_senior,
                          model=model_senior,
                          spez_list=output_router_list).choices[0].message.content
    verbose=False
    if verbose: print(f"\n{bcolors.BGMAGENTA}senior: {bcolors.ENDC} {output_senior}", )
    #12. Запускаем Стилиста:
    output_stilist = stilizator_answer(
                          name=name_stilist,
                          system=system_prompt_stilist,
                          instructions=instructions_stilist,
                          answers_content=output_senior,
                          temp=temperature_stilist,
                          verbose=verbose_stilist,
                          model=model_stilist).choices[0].message.content

    #13. контрольный выстрел по приветствиям:
    output_stilist_withouthello = del_hello(
                          name=name_stilist,
                          system=system_prompt_stilist1,
                          instructions=instructions_prompt_stilist1,
                          answers_content=output_stilist,
                          temp=temperature_stilist,
                          verbose=verbose_stilist,
                          model=model_stilist).choices[0].message.content

    return output_stilist_withouthello

"""# Подгружаем ключ API, Базу знаний

"""

import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
openai.api_key = os.environ["OPENAI_API_KEY"]

# @title Загружаем Базу Знаний, индексируем, сохраняем индексы


with open (r"C:\bakery_bot\shop_catalog.txt", "r", encoding="utf-8") as f:
    knowledge_db_txt = f.read()

# Разбиваем на чанки
docs = split_text(knowledge_db_txt, chunk_size, chunk_overlap)
vectordb = FAISS.from_documents(docs, OpenAIEmbeddings())
vectordb.save_local('UII_database_markdown')



"""# Диалог"""

verbose = 1

#@title Запуск диалога
history_chat = []
history_user = []
history_manager = []

needs_extractor = []
benefits_extractor = []
objection_detector = []
resolved_objection_detector = []
tariff_detector = []

main_answer = ''
summarized_dialog = ''

while True:
    client_question = input("")
    history_user.append(client_question)
    history_chat.append(f"Клиент: {client_question}")
    if len(history_user) == 1: hello_word = sufler(history_user)  # запомнили приветствие
    if client_question.lower() in ['stop', 'стоп']:
        break
    without_hello = get_seller_answer(history_user, history_manager, history_chat)
    if len(history_chat) == 1 and 'None' not in hello_word:
        main_answer = f'{hello_word} שמי אילן אני מוכר במאפיית שלמה.אפשר לשאול מה שמך? ' + without_hello
    else:
        main_answer = without_hello

    #print(f'{bcolors.BGGREEN}Василий:{bcolors.ENDC}\n {insert_newlines(remove_newlines(main_answer), 160)}')
    print((main_answer))

    history_chat.append(f"Менеджер: {without_hello}")
    history_manager.append(without_hello) # не будем Василия хранить в истории, чтобы не путать gpt лишними именами

# Запуск сохранения
"""from datetime import datetime
text_file = f'dialog_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.txt'
with open(text_file, "w") as f:
    f.write(str('\n\n'.join(history_chat)))
"""
