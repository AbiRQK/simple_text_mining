import math
from sklearn.decomposition import pca
from sklearn.cluster.k_means_ import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import random as rnd
import xlrd
import xlsxwriter as xlwr


class TextMining(object):
    def __init__(self, excel_path):
        # Config
        self.distance_method = 'Cosine'  # 'Euclidean', 'Cosine'
        self.k = 5
        self.PCA_feature_extraction = True

        #
        self.cluster_number = list()

        # read information from excel
        self.ds = xlrd.open_workbook(excel_path).sheet_by_name('Sheet1')
        e2e = Export2Excel()

        # Get feature list from abstracts
        self.attribute_dict = dict()
        self.attribute_in_doc = dict()
        self.get_text_features()
        e2e.attribute_list(self.attribute_dict, self.attribute_in_doc)
        # select Features by minimum frequency equal to 3
        self.attributes = self.select_features_by_min_frequency(3)

        # Get information from each articles
        self.articles = list()
        self.get_article_inf()
        # Export matrix to excel
        e2e.tf_idf_matrix(self.articles, self.attributes)

        # # clustering by k_means for finding cluster number
        self.k_means_clustering(self.feature_extraction())
        e2e.sil_score(self.cluster_number)

        # Fuzzy clustering by c-means,
        # matrix is feature extraction result, k in number of cluster
        self.membership_function = list()
        self.c_means_clustering(self.feature_extraction(), self.k)

        # Information aggregation
        e2e.clustering(self.inf_aggregation())

        # write ended excel file
        e2e.write_file()

    def get_text_features(self):
        for i in range(1, self.ds.nrows):
            tokens = PreProcessing(self.ds.cell_value(i, 1)).tokens
            for token in tokens:
                if list(self.attribute_dict.keys()).count(token) == 0:
                    self.attribute_dict[token] = 1
                else:
                    self.attribute_dict[token] = 1 + self.attribute_dict.get(token)

            # Number of documents that use from each feature
            tokens = list(set(tokens))
            for token in tokens:
                if list(self.attribute_in_doc.keys()).count(token) == 0:
                    self.attribute_in_doc[token] = 1
                else:
                    self.attribute_in_doc[token] = 1 + self.attribute_in_doc.get(token)

        return self

    def select_features_by_min_frequency(self, minimum_frequency):
        temp = list()
        for key in self.attribute_dict.keys():
            if self.attribute_dict.get(key) > minimum_frequency:
                temp.append(key)

        return temp

    def doc_number_contain_feature(self):
        temp = dict()
        for f in self.attributes:
            temp[f] = 0

    def get_article_inf(self):
        for i in range(1, self.ds.nrows):
            temp = Article()
            temp.title = self.ds.cell_value(i, 0)
            temp.abstract = self.ds.cell_value(i, 2)
            temp.year = int(self.ds.cell_value(i, 3))
            temp.journal = self.ds.cell_value(i, 4)
            temp.term_frequency, temp.tfidf = self.calc_tf_idf(self.ds.cell_value(i, 1))

            self.articles.append(temp)

    def calc_tf_idf(self, string):
        tf_vec = list()
        tfidf_vec = list()
        vec = PreProcessing(string).tokens
        for i in range(len(self.attributes)):
            tf = vec.count(self.attributes[i])
            tf_vec.append(tf)

            # TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
            tf = tf / len(vec)
            # IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
            idf = math.log10((self.ds.nrows - 1) / self.attribute_in_doc.get(self.attributes[i]))
            tfidf_vec.append(tf * idf)

        return tf_vec, tfidf_vec

    def feature_extraction(self):
        mat = list()
        for t in self.articles:
            mat.append(t.tfidf)

        # Feature extraction using PCA
        if self.PCA_feature_extraction:
            pca1 = pca.PCA(n_components=len(mat))
            pca1.fit(mat)
            mat = pca1.components_
            mat = [x[0:10] for x in mat]

        return mat

    def k_means_clustering(self, matrix):
        for k in range(3, 10):
            km = KMeans(n_clusters=k)
            self.cluster_number.append([k, silhouette_score(matrix, km.fit_predict(matrix))])

        return self

    def c_means_clustering(self, mat, cluster):
        fuzzifier = 2  # fuzzy factor is bigger than 1 (equal to 1 is crisp)
        fcm = 0
        # Randomly initialize the membership matrix
        membership = list()
        dim = len(mat[0])
        for i in range(len(mat)):
            temp = list()
            for j in range(cluster):
                temp.append(rnd.randint(1, 10))
            membership.append([x / sum(temp) for x in temp])

        for iter in range(100):
            # Calculate the centroid using equation
            centroid = list()
            for k in range(cluster):
                w2 = [math.pow(w[k], fuzzifier) for w in membership]
                temp = list()
                for d in range(dim):
                    temp.append(sum([mat[i][d] * w2[i] for i in range(len(mat))]) / sum(w2))

                centroid.append(temp)

            # calculate similarity
            membership = list()
            for i in range(len(mat)):
                temp = list()
                for j in range(len(centroid)):
                    dist = self.calc_distance(mat[i], centroid[j])
                    if dist == 0:
                        temp.append(1)
                    else:
                        t1 = [self.calc_distance(mat[i], c) for c in centroid]
                        t2 = [dist / x for x in t1]
                        temp.append(1 / sum([math.pow(x, 2 / (fuzzifier - 1)) for x in t2]))

                membership.append(temp)

            # fcm
            fcm_new = 0
            for i in range(len(mat)):
                for j in range(len(centroid)):
                    fcm_new += (math.pow(membership[i][j], 1)) * (distance.euclidean(mat[i], centroid[j]))

            if abs(fcm_new - fcm) < 0.0001:
                break
            else:
                fcm = fcm_new

        self.membership_function = membership
        return self

    def calc_distance(self, x, y):
            dis = 0
            if self.distance_method == 'Euclidean':
                dis = distance.euclidean(x, y)
            elif self.distance_method == 'Cosine':
                dis = distance.cosine(x, y)

            return dis

    def inf_aggregation(self):
        inf = list()
        for i in range(len(self.articles)):
            art = self.articles[i]
            temp = [art.title, art.year, art.journal]
            for m in self.membership_function[i]:
                temp.append(m)
            inf.append(temp)

        return inf


class Article(object):
    def __init__(self):
        self.title = str()
        self.key_word = str()
        self.year = int()
        self.journal = str()
        self.abstract = str()
        self.term_frequency = list()
        self.tfidf = list()
        self.cluster = int()
        self.cluster_membership = list()


class PreProcessing(object):
    def __init__(self, string=str()):
        self.string = string
        self.tokens = self.clean_string().split()
        self.remove_stop_word()

    def clean_string(self):
        temp_ = self.string.lower()
        # remove character
        for ch in self.character():
            temp_ = temp_.replace(ch, '')

        for numb in self.numbers():
            temp_ = temp_.replace(numb, '')

        return temp_

    def remove_stop_word(self):
        for sw in self.stop_word():
            for a in range(self.tokens.count(sw)):
                self.tokens.remove(sw)

        return self

    @staticmethod
    def stop_word():
        # return english stop word
        return {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
                'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don',
                'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
                'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
                'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
                'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

    @staticmethod
    def character():
        return {'.', ',', ':', ';', '(', ')', '·', '/', '?', '!', '"', '[', ']', '`', '\'',
                '“', '”', '’', '@', '&', '‘'}

    @staticmethod
    def numbers():
        return {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}


class Export2Excel(object):
    def __init__(self):
        self.file = xlwr.Workbook('TextMiningResult.xlsx')
        self.att_list_sheet = self.file.add_worksheet('AttributeList')
        self.tf_sheet = self.file.add_worksheet('TermFrequency')
        self.tfidf_sheet = self.file.add_worksheet('TFIDF')
        self.svd_sheet = self.file.add_worksheet('SVD')
        self.silhouette_score_sheet = self.file.add_worksheet('SilhouetteScoreClusterNumber')
        self.clustering_sheet = self.file.add_worksheet('Fuzzy clustering')

    def write_file(self):
        self.file.close()

    def clustering(self, mat):
        self.clustering_sheet.write(0, 0, 'Title')
        self.clustering_sheet.write(0, 1, 'year')
        self.clustering_sheet.write(0, 2, 'journal')
        self.clustering_sheet.write(0, 3, 'Cluster0')

        i = 1
        for m in mat:
            for j in range(len(m)):
                self.clustering_sheet.write(i, j, m[j])
            i += 1

    def sil_score(self, cluster_score):
        self.silhouette_score_sheet.write(0, 0, 'ClusterNumber')
        self.silhouette_score_sheet.write(0, 1, 'SilhouetteScore')
        i = 1
        for cs in cluster_score:
            self.silhouette_score_sheet.write(i, 0, cs[0])
            self.silhouette_score_sheet.write(i, 1, cs[1])
            i += 1

    def attribute_list(self, att_dict, attribute_in_doc):
        self.att_list_sheet.write(0, 0, 'Attributes')
        self.att_list_sheet.write(0, 1, 'frequency')
        self.att_list_sheet.write(0, 2, 'doc_frequency')

        i = 1
        for key in att_dict.keys():
            self.att_list_sheet.write(i, 0, key)
            self.att_list_sheet.write(i, 1, att_dict.get(key))
            self.att_list_sheet.write(i, 2, attribute_in_doc.get(key))
            i += 1

    def tf_idf_matrix(self, article, attributes):
        self.tf_sheet.write(0, 0, 'Title')
        self.tfidf_sheet.write(0, 0, 'Title')
        j = 1
        for att in attributes:
            self.tf_sheet.write(0, j, att)
            self.tfidf_sheet.write(0, j, att)
            j += 1

        i = 1
        for t in article:
            self.tf_sheet.write(i, 0, t.title)
            self.tfidf_sheet.write(i, 0, t.title)
            for j in range(len(attributes)):
                self.tf_sheet.write(i, j + 1, t.term_frequency[j])
                self.tfidf_sheet.write(i, j + 1, t.tfidf[j])
            i += 1
        return self

    def svd_matrix(self):
        pass


# Run
TextMining('TextMiningDS.xlsx')
