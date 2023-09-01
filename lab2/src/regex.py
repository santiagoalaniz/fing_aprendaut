date_pattern = r'\d{1,2}/\d{1,2}/\d{2,4}'
time_pattern = r'\d{1,2}:\d{1,2}'
user_pattern = r'[\w\s\+\-]+'
metadata_pattern = f'({date_pattern}, {time_pattern} - {user_pattern}: )'
message_pattern = r'([\s\S]+?)'

LOG_ENTRY_PATTERN = f'{metadata_pattern}{message_pattern}(?=\n{date_pattern}, {time_pattern} - |$)'
