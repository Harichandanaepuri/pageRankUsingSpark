import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object TopicModelling {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: TopicModelling InputDir OutputDir")
    }
    val  spark = SparkSession
      .builder()
      .appName("Topic Modelling")
      .config("spark.master", "local")
      .getOrCreate()

    val book = spark.read.textFile(args(0)).toDF("text")
    //val book = spark.read.textFile("C://Users//haric//Downloads//Assignment2//src//main//Resources//60472-0.txt").toDF("text")
    //    for (elem <- book.take(10)) {println(elem)}
    val tkn = new Tokenizer().setInputCol("text").setOutputCol("textOut")
    val tokenized = tkn.transform(book)
    val remover = new StopWordsRemover()
      .setInputCol("textOut")
      .setOutputCol("filtered")
    val filtered_df = remover.transform(tokenized)


    val cv = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(10000)
      .setMinTF(0)
      .setMinDF(0)
      .setBinary(true)
    val cvFitted = cv.fit(filtered_df)
    val prepped = cvFitted.transform(filtered_df)
    val lda = new LDA().setK(5).setMaxIter(5)
    //println(lda.explainParams())
    val model = lda.fit(prepped)


    val vocabList = cvFitted.vocabulary
    val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }
    val topics = model.describeTopics(maxTermsPerTopic = 5)
      .withColumn("terms", termsIdx2Str(col("termIndices")))
    val results = topics.select("topic", "terms", "termWeights")
    results.show()
    results.rdd.map(_.toString()).saveAsTextFile(args(1))
    //for (e <- results.take(3)) {println(e)}
  }
}
