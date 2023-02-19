import sys
import copy    # for deepcopy()
import string

Debug = False   # Sometimes, print for debugging
InputFilename = "data.v1"
TargetWords = [
        'outside', 'today', 'weather', 'raining', 'nice', 'rain', 'snow',
        'day', 'winter', 'cold', 'warm', 'snowing', 'out', 'hope', 'boots',
        'sunny', 'windy', 'coming', 'perfect', 'need', 'sun', 'on', 'was',
        '-40', 'jackets', 'wish', 'fog', 'pretty', 'summer'
        ]


def open_file(filename=InputFilename):
    try:
        f = open(filename, "r")
        return(f)
    except FileNotFoundError:
        # FileNotFoundError is subclass of OSError
        if Debug:
            print("File Not Found")
        return(sys.stdin)
    except OSError:
        if Debug:
            print("Other OS Error")
        return(sys.stdin)


def safe_input(f=None, prompt=""):
    try:
        # Case:  Stdin
        if f is sys.stdin or f is None:
            line = input(prompt)
        # Case:  From file
        else:
            assert not (f is None)
            assert (f is not None)
            line = f.readline()
            if Debug:
                print("readline: ", line, end='')
            if line == "":  # Check EOF before strip()
                if Debug:
                    print("EOF")
                return("", False)
        return(line.strip(), True)
    except EOFError:
        return("", False)


class C274:
    def __init__(self):
        self.type = str(self.__class__)
        return

    def __str__(self):
        return(self.type)

    def __repr__(self):
        s = "<%d> %s" % (id(self), self.type)
        return(s)


class ClassifyByTarget(C274):
    def __init__(self, lw=[]):
        # FIXME:  Call superclass, here and for all classes
        self.type = str(self.__class__)
        self.allWords = 0
        self.theCount = 0
        self.nonTarget = []
        self.set_target_words(lw)
        self.initTF()
        return

    def initTF(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        return

    def get_TF(self):
        return(self.TP, self.FP, self.TN, self.FN)

    # FIXME:  Use Python properties
    #     https://www.python-course.eu/python3_properties.php
    def set_target_words(self, lw):
        # Could also do self.targetWords = lw.copy().  Thanks, TA Jason Cannon
        self.targetWords = copy.deepcopy(lw)
        return

    def get_target_words(self):
        return(self.targetWords)

    def get_allWords(self):
        return(self.allWords)

    def incr_allWords(self):
        self.allWords += 1
        return

    def get_theCount(self):
        return(self.theCount)

    def incr_theCount(self):
        self.theCount += 1
        return

    def get_nonTarget(self):
        return(self.nonTarget)

    def add_nonTarget(self, w):
        self.nonTarget.append(w)
        return

    def print_config(self):
        print("-------- Print Config --------")
        ln = len(self.get_target_words())
        print("TargetWords Hardcoded (%d): " % ln, end='')
        print(self.get_target_words())
        return

    def print_run_info(self):
        print("-------- Print Run Info --------")
        print("All words:%3s. " % self.get_allWords(), end='')
        print(" Target words:%3s" % self.get_theCount())
        print("Non-Target words (%d): " % len(self.get_nonTarget()), end='')
        print(self.get_nonTarget())
        return

    def print_confusion_matrix(self, targetLabel, doKey=False, tag=""):
        assert (self.TP + self.TP + self.FP + self.TN) > 0
        print(tag+"-------- Confusion Matrix --------")
        print(tag+"%10s | %13s" % ('Predict', 'Label'))
        print(tag+"-----------+----------------------")
        print(tag+"%10s | %10s %10s" % (' ', targetLabel, 'not'))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'TP   ', 'FP   '))
        print(tag+"%10s | %10d %10d" % (targetLabel, self.TP, self.FP))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'FN   ', 'TN   '))
        print(tag+"%10s | %10d %10d" % ('not', self.FN, self.TN))
        return

    def eval_training_set(self, tset, targetLabel):
        print("-------- Evaluate Training Set --------")
        self.initTF()
        z = zip(tset.get_instances(), tset.get_lines())
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class()
            if lb == targetLabel:
                if cl:
                    self.TP += 1
                    outcome = "TP"
                else:
                    self.FN += 1
                    outcome = "FN"
            else:
                if cl:
                    self.FP += 1
                    outcome = "FP"
                else:
                    self.TN += 1
                    outcome = "TN"
            explain = ti.get_explain()
            print("TW %s: ( %10s) %s" % (outcome, explain, w))
            if Debug:
                print("-->", ti.get_words())
        self.print_confusion_matrix(targetLabel)
        return

    def classify_by_words(self, ti, update=False, tlabel="last"):
        inClass = False
        evidence = ''
        lw = ti.get_words()
        for w in lw:
            if update:
                self.incr_allWords()
            if w in self.get_target_words():    # FIXME Write predicate
                inClass = True
                if update:
                    self.incr_theCount()
                if evidence == '':
                    evidence = w            # FIXME Use first word, but change
            elif w != '':
                if update and (w not in self.get_nonTarget()):
                    self.add_nonTarget(w)
        if evidence == '':
            evidence = '#negative'
        if update:
            ti.set_class(inClass, tlabel, evidence)
        return(inClass, evidence)

    # Could use a decorator, but not now
    def classify(self, ti, update=False, tlabel="last"):
        cl, e = self.classify_by_words(ti, update, tlabel)
        return(cl, e)

class ClassifyByTopN(ClassifyByTarget):
    def __init__(self,lw=[]):
        self.type=str(self.__class__)
        super().__init__(lw)
        return

    def target_top_n(self,tset,num=5,label=''):
        from collections import Counter
        self.num=num
        self.label=label
        random_dict_01=Counter()
        is_preprocessed=hasattr(TrainingSet,"random_list_03")

        if is_preprocessed==True:
            getattr(TrainingSet,"random_list_03")
            random_dict_01=Counter(random_list_03)
        else:

            for i in tset.get_instances():
                if i.get_label()==self.label:
                    words01=i.get_words()
                    word_count_dict=Counter(words01)
                    random_dict_01+=word_count_dict

        random_list_04=sorted(random_dict_01.items(),key=lambda x:x[1],reverse=True)
        random_list_05=[]
        random_list_06=[]

        for j in random_list_04:
            random_list_05.append(j[1])

        for k in random_list_05:
            if k in random_list_06:
                pass
            else:
                random_list_06.append(k)

        top_numbers=random_list_06[:self.num]

        new_target_words=[]
        for l in random_dict_01: #searching for the top words
            if random_dict_01[l] in top_numbers:
                new_target_words.append(l)
        if new_target_words[0]==self.label:
            new_target_words.pop(0)
        else:
            pass
        self.targetWords=new_target_words
        return





class TrainingInstance(C274):
    def __init__(self):
        self.type = str(self.__class__)
        self.inst = dict()
        # FIXME:  Get rid of dict, and use attributes
        self.inst["label"] = "N/A"      # Class, given by oracle
        self.inst["words"] = []         # Bag of words
        self.inst["class"] = ""         # Class, by classifier
        self.inst["explain"] = ""       # Explanation for classification
        self.inst["experiments"] = dict()   # Previous classifier runs
        return

    def get_label(self):
        return(self.inst["label"])

    def get_words(self):
        return(self.inst["words"])

    def set_class(self, theClass, tlabel="last", explain=""):
        # tlabel = tag label
        self.inst["class"] = theClass
        self.inst["experiments"][tlabel] = theClass
        self.inst["explain"] = explain
        return

    def get_class_by_tag(self, tlabel):             # tlabel = tag label
        cl = self.inst["experiments"].get(tlabel)
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_explain(self):
        cl = self.inst.get("explain")
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_class(self):
        return self.inst["class"]

    def process_input_line(
                self, line, run=None,
                tlabel="read", inclLabel=True
            ):
        for w in line.split():
            if w[0] == "#":
                self.inst["label"] = w
                # FIXME: For testing only.  Compare to previous version.
                if inclLabel:
                    self.inst["words"].append(w)
            else:
                self.inst["words"].append(w)

        if not (run is None):
            cl, e = run.classify(self, update=True, tlabel=tlabel)
        return(self)

    def preprocess_words(self,mode=''):
        self.preprocessed_list=[]
        self.mode=mode
        task_01_abc=self.get_words()
        if self.mode=='keep-digits':
            x=self.keep_digits(task_01_abc)
            self.inObjHash=self.remove_stop_words(x)
        elif self.mode=='keep-stops':
            y=self.keep_digits(task_01_abc)
            self.inObjHash=self.remove_digits(y)
            #print(*z)
        elif self.mode=='keep-symbols':
            w=self.remove_digits(task_01_abc)
            self.inObjHash=self.remove_stop_words(w)
        else:
            x=self.keep_digits(task_01_abc)
            y=self.remove_digits(x)
            self.preprocessed_list.append(self.remove_stop_words(y))
        self.inst["words"]=self.preprocessed_list[0]
        return(self.inst["words"])
        

    def keep_digits(self,a):

        text_02=[]
        # for loop iterating over the list that was inputted to the function 
        # and removing all punctutaions, and then appending the new results to another list
        for o in a:
            result=""
            for p in o:
                if p in string.ascii_lowercase or p.isnumeric():
                    result+=p
            text_02.append(result)
        return(text_02)


    def remove_digits(self,b):

        text_03=[]
        # for loop iterating over the input list to remove the numbers that are attached to any other character
        # and then appends the results to a new list and returns it.
        for k in b:
            results=""
            if k.isnumeric()==False:
                for l in k:
                    if l.isnumeric()==False:
                        results+=l
                text_03.append(results)
            else:
                text_03.append(k)
        return(text_03)

    def remove_stop_words(self,a):
        """ This function takes in a list and checks if the words in the list are stop-words or not,
            if they are, it removes them otherwise appends them to a new list "processed_words" 
            and prints the new list """

        stop_words= ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your","yours", "yourself", "yourselves", "he",
        "him", "his", "himself", "she", "her","hers",
        "herself", "it", "its", "itself", "they", "them",
        "their", "theirs","themselves", "what", "which","who",
        "whom", "this", "that", "these", "those","am", "is", "are",
        "was", "were", "be","been", "being", "have", "has", "had","having",
        "do", "does", "did", "doing", "a", "an","the", "and", "but", "if","or",
        "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before", 
        "after","above", "below", "to", "from", "up", "down", "in", "out", "on",
        "off", "over","under", "again", "further", "then", "once", "here", "there",
        "when", "where","why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other","some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than","too", "very", "s", "t", "can", "will", "just", "don",
        "should", "now"]
        processed_words=[]
        # for loop iterating over the inputted list to remove all the stop words and 
        # appending and printing all the other words.
        for m in a:
            if m not in stop_words:
                processed_words.append(m)
        return(processed_words)
           



class TrainingSet(C274):
    def __init__(self):
        self.type = str(self.__class__)
        self.inObjList = []     # Unparsed lines, from training set
        self.inObjHash = []     # Parsed lines, in dictionary/hash
        self.random_list_03=[]
        return

    def get_instances(self):
        return(self.inObjHash)      # FIXME Should protect this more

    def get_lines(self):
        return(self.inObjList)      # FIXME Should protect this more

    def print_training_set(self):
        print("-------- Print Training Set --------")
        z = zip(self.inObjHash, self.inObjList)
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class_by_tag("last")     # Not used
            explain = ti.get_explain()
            print("( %s) (%s) %s" % (lb, explain, w))
            if Debug:
                print("-->", ti.get_words())
        return

    def process_input_stream(self, inFile, run=None):
        assert not (inFile is None), "Assume valid file object"
        cFlag = True
        while cFlag:
            line, cFlag = safe_input(inFile)
            if not cFlag:
                break
            assert cFlag, "Assume valid input hereafter"

            # Check for comments
            if line[0] == '%':  # Comments must start with %
                continue

            # Save the training data input, by line
            self.inObjList.append(line)
            # Save the training data input, after parsing
            ti = TrainingInstance()
            ti.process_input_line(line, run=run)
            self.inObjHash.append(ti)
        return

    def preprocess(self,mode=''):
        for i in self.inObjHash: #for loop iterating over inobjhash and each time calls preprocess_words
            i.preprocess_words() # from TrainingInstance



    def return_nfolds(self,num=3):
        
        divide_n_folds=[]
        for initial_01 in range(num):
            divide_n_folds.append(TrainingSet()) #passing the training set which has to be divided into n folds 
        initial_01=0
        for i in self.get_lines(): #for loop iterating over inobjlist 
            divide_n_folds[initial_01].inObjList.append(copy.deepcopy(i))
            if initial_01==num-1:
                initial_01=0
            else:
                initial_01+=1
        initial_02=0
        for j in self.get_instances(): #for loop iterating over inobjhash
            divide_n_folds[initial_02].inObjHash.append(copy.deepcopy(j))
            if initial_02==num-1:
                initial_02=0
            else:
                initial_02+=1
        return(divide_n_folds)

    def copy(self):
        copy01=copy.deepcopy(self)
        return copy01

    def add_fold(self,tset):
        self.inObjList.extend(copy.deepcopy(tset.inObjList))
        self.inObjHash.extend(copy.deepcopy(tset.inObjHash))
        return







def basemain():                                                     
    tset = TrainingSet()
    run1 = ClassifyByTarget(TargetWords)
    print(run1)     # Just to show __str__
    lr = [run1]
    print(lr)       # Just to show __repr__

    argc = len(sys.argv)
    if argc == 1:   # Use stdin, or default filename
        inFile = open_file()
        assert not (inFile is None), "Assume valid file object"
        tset.process_input_stream(inFile, run1)
        inFile.close()
    else:
        for f in sys.argv[1:]:
            inFile = open_file(f)
            assert not (inFile is None), "Assume valid file object"
            tset.process_input_stream(inFile, run1)
            inFile.close()

    if Debug:
        tset.print_training_set()
    run1.print_config()
    run1.print_run_info()
    run1.eval_training_set(tset, '#weather')
    tset.preprocess()
    run1.target_top_n(tset, num=3, label=plabel)

    
    

    return


if __name__ == "__main__":
    basemain()
