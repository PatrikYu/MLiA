function word_indices = processEmail(email_contents)
%PROCESSEMAIL preprocesses a the body of an email and
%returns a list of word_indices 
%   word_indices = PROCESSEMAIL(email_contents) preprocesses 
%   the body of an email and returns a list of indices of the 
%   words contained in the email. 
%   预处理电子邮件的正文并返回电子邮件中包含的单词的索引列表。

% Load Vocabulary  获取单词表的结构体文件
vocabList = getVocabList();

% Init return value 初始化返回值为空矩阵
word_indices = [];

% ========================== Preprocess Email  预处理电子邮件 ===========================

% Find the Headers 找到标题 ( \n\n and remove )
% Uncomment the following lines if you are working with raw emails with the
% full headers
% 如果您使用完整标题的原始电子邮件，请取消注释以下行

% hdrstart = strfind(email_contents, ([char(10) char(10)]));
% email_contents = email_contents(hdrstart(1):end);

% Lower case  把内容转换为小写
email_contents = lower(email_contents);

% Strip all HTML 去除所有的HTML
% Looks for any expression that starts with < and ends with > and replace
% and does not have any < or > in the tag it with a space
% 查找任何以<开头并以＆>结尾的表达式，并替换，并且在标签中没有任何<or>，并带有空格
% regexprep(str,expression,replace) 此处用‘ ’来替换html
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

% Handle Numbers 处理数字
% Look for one or more characters between 0-9
email_contents = regexprep(email_contents, '[0-9]+', 'number');

% Handle URLS 处理链接
% Look for strings starting with http:// or https://
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');

% Handle Email Addresses
% Look for strings with @ in the middle
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

% Handle $ sign 处理$这个标志
email_contents = regexprep(email_contents, '[$]+', 'dollar');


% ========================== Tokenize Email ===========================

% Output the email to screen as well  将电子邮件输出到屏幕
fprintf('\n==== Processed Email ====\n\n');

% Process file 处理文件
l = 0;

while ~isempty(email_contents)  % 当邮件内容不为空时

    % Tokenize and also get rid of any punctuation（去除标点符号）
    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
   
    % Remove any non alphanumeric characters 删除任何非字母数字字符
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word 
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try str = porterStemmer(strtrim(str)); 
    catch str = ''; continue;
    end;

    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end

    % Look up the word in the dictionary and add to word_indices if
    % found    在字典中查找单词并添加到word_indices（如果找到）
    % ====================== YOUR CODE HERE ======================
    % Instructions: Fill in this function to add the index of str to
    %               word_indices if it is in the vocabulary. At this point
    %               of the code, you have a stemmed word from the email in
    %               the variable str. You should look up str in the
    %               vocabulary list (vocabList). If a match exists, you
    %               should add the index of the word to the word_indices
    %               vector. Concretely, if str = 'action', then you should
    %               look up the vocabulary list to find where in vocabList
    %               'action' appears. For example, if vocabList{18} =
    %               'action', then, you should add 18 to the word_indices 
    %               vector (e.g., word_indices = [word_indices ; 18]; ).
    % 
    % Note: vocabList{idx} returns a the word with index idx in the
    %       vocabulary list. 
    % vocabList {idx}在词汇列表中返回索引为idx的单词。
    % 
    % Note: You can use strcmp(str1, str2) to compare two strings (str1 and
    %       str2). It will return 1 only if the two strings are equivalent.
    %       只有两个字符串相同时它才会返回1。
    vocabSize = size(vocabList);
    
    for index = 1 : vocabSize,
    	if strcmp(vocabList{index} , str),
    		word_indices = [word_indices ; index];   %将存在的字符的索引填入矩阵
    	end
    end
    
   

    % =============================================================


    % Print to screen, ensuring that the output lines are not too long
    % 打印到屏幕上，确保一行输出不会太长
    if (l + length(str) + 1) > 78
        fprintf('\n');
        l = 0;
    end
    fprintf('%s ', str);
    l = l + length(str) + 1;

end

% Print footer
fprintf('\n\n=========================\n');

end
