from nltk.tokenize import TweetTokenizer

class Tokenizer :
    '''
    This class provides a tokenizer which can be used to tokenize tweets
    into words or into chararacters.
    '''

    def __init__ (self, token_level="word"):
        self.token_level = token_level
        self.nltk_tokenizer = TweetTokenizer(
            preserve_case=True,
            reduce_len=True,
            strip_handles=True )
    
    def tokenize(self, tweet):
        '''
        Uses the nltk TweetTokenizer in order to achieve a first segmentation
        of the tweet then, if specified, segments the tokens again to achieve
        character level tokenization.
        '''
        tokens = self.nltk_tokenizer.tokenize(tweet)
        if self.token_level == "char":
            char_tokens = []
            for word in tokens:
                char_tokens += [char for char in word+" "]
            tokens = char_tokens[:-1]
        return tokens
