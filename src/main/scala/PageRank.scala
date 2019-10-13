import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}

object PageRank {
  def main(args: Array[String]): Unit = {
//    if(args.length!=3) {
//      print("3 arguments are required (input file, no of iterations, output file")
//    }
    val conf = new SparkConf().setAppName("PageRank").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    //val input = sc.textFile("C://Users//haric//Downloads//Assignment2//src//main//Resources//87863972_T_ONTIME_REPORTING.csv")
    val input = sc.textFile(args(0))
    val header = input.first
    val data = input.filter(line => line != header)
    val Origin_Destination = data.map(line => ((line.split(","))(0), (line.split(","))(3)))
    val N = Origin_Destination.count().toDouble
    print("links = "+N)
    //val links = sc.parallelize(List(("MapR",List("Baidu","Blogger")),("Baidu", List("MapR")),("Blogger",List("Google","Baidu")),("Google", List("MapR")))).partitionBy(new HashPartitioner(4)).persist()
    val links = Origin_Destination.map(s => (s._1.toString,s._2.toString))
      .groupBy(_._1)
      .mapValues(_.map(_._2))
    val alpha = 0.15
    val iterations = args(1).toInt
    var pageRanks = links.mapValues(x => 1.0)
    var i=0
    for(i <- 1 to iterations) {
      println("--------------------------------------------")
      pageRanks.collect().foreach(println)
      val sum_cont_by_each_inlink = links.join(pageRanks).flatMap { case (url, (links, rank)) => links.map(node => (node, rank / links.size)) }
      val newRanks = sum_cont_by_each_inlink.reduceByKey((x, y) => x + y).mapValues(line => alpha/N + (1-alpha) * line)
      pageRanks = newRanks
    }
    val sorted_by_rank = pageRanks.sortBy(-_._2)
    sorted_by_rank.collect().foreach(println)
    sorted_by_rank.saveAsTextFile(args(2))
  }
}