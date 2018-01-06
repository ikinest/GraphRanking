/**
  * author:ikinest(shijp)
  * Created by April on 2017/5/29.
  */

import java.io.{FileWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks


//define a interface for vertex of different type in bipartite graph
trait VertexProperty
case class UserProperty(val name:String) extends VertexProperty
case class ProductProperty(val name:String)extends VertexProperty
case class PositiveProperty(val name:String)extends VertexProperty



object graphRS {

  //construct positive rating
  def positiveRDD(avgScore:RDD[(VertexId,Double)],edges: RDD[Edge[Double]])
  :RDD[Edge[Double]]={
    //avgScore.map(s=>s._1)
    avgScore.map(s=>(s._1,(s._2))).join(edges.map(e=>(e.srcId,(e.dstId,e
      .attr)))).filter(f=>f._2._1 <= f._2._2._2).map(m=>Edge(m._1,m._2._2._1,
      m._2._2._2))

  }

  def personalizedPositeItem(positiverdd:RDD[Edge[Double]]):RDD[(VertexId, Set[VertexId])]={
    positiverdd.map(p=>(p.srcId,Set(p.dstId))).reduceByKey(_++_)

  }


  def personlizedPreferenceItem(personlizedPostive:RDD[(Set
    [VertexId])])={
    personlizedPostive.map(_.subsets(2)
      .map(_.toList).toList)
      .flatMap{case innerList => innerList.map(_ -> 1.0) }
      .map(in=>(((in._1(0),in._1(1)),in._2)))
      .reduceByKey(_+_)
      .map(e=>Edge(e._1._1,e._1._2,e._2))
      .sortBy(_.attr)

  }

  def PernonalizedSubGraph(user:Int,positiveItem:RDD[(VertexId,
    Set[VertexId])],gg:Graph[Int,Double])={
    val filteruser = positiveItem.filter(_._1.toInt == user)
    val set =  filteruser.map(u=>u._2).collect()
    (user,gg.subgraph(vpred = (vid,v)=>v.toString
      .nonEmpty,
      epred =
        edge=>(set(0)
          .contains(edge.srcId))))
  }

  def PernonalizedSubGraph_RDD(user:RDD[(VertexId,String)],positiveItem:RDD[
    (VertexId,
      Set[VertexId])],gg:Graph[Int,Double])={
    // have some problems to solve later
    val set =   positiveItem.join(user).filter(x=>x._2._2 == "recom")
      .map(m=>(m._1,(gg.subgraph(vpred = (vid,v)=>v.toString.nonEmpty,epred
        = edge=>(m._2._1.contains(edge.srcId))))))
    //      positiveItem.filter(_._1.toInt == user).map(u=>u._2).collect()
    //    gg.subgraph(vpred = (vid,v)=>v.toString.nonEmpty,epred = edge=>(set(0).contains
    //    (edge.srcId)))

  }

  //  def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]) =
  //    sum(pow(v1 - v2, 2))

  def getMovieCluster(rawrating:RDD[(String,String,String,
    String)])={
    val ratings = rawrating.map{case(user,movie,rating,_)=>Rating(user.toInt,
      movie.toInt,rating.toDouble)}
    ratings.cache()
    val alsModel = ALS.train(ratings,10,10,0.1)
    val movieFactor = alsModel.productFeatures.map{case (id,factor)=>(id,
      Vectors.dense(factor))}
    val movieVectors = movieFactor.map(_._2).cache()
    val movieClusterModel = KMeans.train(movieVectors,100,50,10)
    val movieAssigned = movieFactor.map{
      case (id,vector)=>
        val pred = movieClusterModel.predict(vector)
        val clusterCentre = movieClusterModel.predict(vector)
        //val dist = computeDistance(DenseVector(clusterCentre.toInt),
        //DenseVector(vector.toArray))
        (id,pred)
    }

    movieAssigned.map(m=>(m._1,m._2))
  }


  def CacuDiversity(personalgWithUser:(Int,Graph[Int,Double]))={

    val user = personalgWithUser._1
    val personalg = personalgWithUser._2
    val trip = personalg.triplets.map(t=>(t.srcId,(t.dstId,t.attr)))++(personalg
      .triplets
      .map(tt=>(tt.dstId,(tt.srcId,tt.attr)))).cache()

    //(srcId,((dstId,(srcid,weightatrr)),srcdegree))
    val srcIdDegree = trip.map(r=>(r._2._1,(r._1,(r._2._1,r._2._2))))
      .join(personalg.degrees).cache()

    //(dstid, (srcid, (srcid,weightattr)))
    val srcidTriplet =
      srcIdDegree.map(tp=>tp._2)
        .map(tp2=>(tp2._1,tp2._2))
        .map(tp3=>(tp3._1._1,(tp3._1._2._1,(tp3._1._2._1,tp3._1._2
          ._2)))).cache()

    //    (dstId,((dstId,srcId,weightattr), dstdegree)))
    val dstIdDegree =
      srcidTriplet.map(dst=>
        (dst._2._1,(dst._2._2._1,dst._1,dst._2._2._2))
      ).join(personalg.degrees).cache()

    //(dstid, (srcid, (srcid,weightattr)))
    val entropy =
      srcidTriplet.map(src=>((src._1,src._2._1),(src._2._2._2)))
        .join(
          dstIdDegree.map(dst=>((dst._1,dst._2._1._2),(dst._2._2)))
        )
        .map(en=>(en._1._2,(en._2._1/en._2._2.toDouble)*log2((en._2._1/en._2._2
          .toDouble))))
        .reduceByKey(_+_)
        .map(pos=>(pos._1,-pos._2))
        .cache()

    val div = entropy.join(personalg.degrees).map(jointly=>(jointly._1,
      jointly._2
        ._1*jointly._2._2)).sortBy(_._2,ascending = false).join(personalg
      .vertices.map(v=>(v._1,v._2))).map(en=>(en._1,en._2._2,en._2._1))
    (user,div)

  }

  def log2(x:Double)= math.log10(x)/math.log10(2.0)


  def recomendation(diversityWithUser:(Int,RDD[(VertexId,Int,Double)]),
                    topn:Int,
                    catWeight:Int
                    =1):(Int,ListBuffer[VertexId]) ={
    val reclist = new ListBuffer[VertexId]
    val catlist = new ListBuffer[Int]
    val user = diversityWithUser._1
    val diversity = diversityWithUser._2
    val loop = new Breaks

    val sortcollect = diversity.sortBy(_._3,ascending = false).collect()
    var start = 0
    while(reclist.length < topn){
      loop.breakable{
        for (len <- start until sortcollect.length) {
          val taker = sortcollect(len)
          if (!catlist.contains(taker._2)) {
            catlist.append(taker._2)
            reclist.append(taker._1)
            //println(111111111)
            start = len
            loop.break

          }
        }
      }
    }
    return (user,reclist)
  }

  def main(args: Array[String]) {
    // avoid some redundant info
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)


    // init sparkcontext   //"spark://master:7077"
    val conf = new SparkConf().setAppName("graphRS")
      .setMaster("local["+args(0)+"]")
    .set("spark.default.parallelism", args(6))
      .set("spark.shuffle.blockTransferService", "nio")
    val sc = new SparkContext(conf)

    // load rating file    test.txt
    val data = sc.textFile(args(1))

    // sparse ratings
    val lines = data.map{line=>
      val fields = line.split(args(2))
      (fields(0),fields(1),fields(2),fields(3))}.cache()

    val cluster = getMovieCluster(lines)

    var startTime = System.currentTimeMillis()

    // construct bipartite graph vertex
    val users:RDD[(VertexId,VertexProperty)] = lines.map(
      line=>(("-"+line._1).toLong,UserProperty("user"+line._1)))
    users.cache()

    val items:RDD[(VertexId,VertexProperty)] = lines.map(
      line=>(line._2.toLong,ProductProperty("item"+line._2)))
    items.cache()

    // construct edges
    val edgesRDD : RDD[Edge[Double]] = lines.map(line=>Edge
    (("-"+line._1).toLong,line._2.toLong,line._3.toDouble))
    edgesRDD.cache()

    // merge different type vertex
    val verticesRDD = VertexRDD(users ++ items)
    verticesRDD.cache()

    // construct bipartite graph
    val g:Graph[VertexProperty,Double]=Graph(verticesRDD,edgesRDD)

    g.persist()


    val totalScoreRDD = g.aggregateMessages[Double](triplet=>{triplet.sendToSrc(triplet.attr)},_ + _,
      TripletFields.EdgeOnly)
    totalScoreRDD.cache()

    val outdegreeRDD = g.outDegrees
    outdegreeRDD.cache()

    val avgScoreRDD:RDD[(VertexId,Double)] =
      totalScoreRDD.innerJoin(outdegreeRDD)(
        (id,a,b)=>a/b).map(
        x=>(x._1,(x._2,PositiveProperty(
          "avgScore"+x._1.toString)))).map(m=>(m._1,m._2._1))
    avgScoreRDD.cache()

    //g.unpersist()


    val positeveEdgeRDD = positiveRDD(avgScoreRDD,edgesRDD)
    positeveEdgeRDD.cache()

    val positeGraph:Graph[VertexProperty,Double]=Graph(verticesRDD,positeveEdgeRDD)

    positeGraph.cache()

    //    positeveEdgeRDD.foreach(println)
    //    println("______________")
    //    positeGraph.vertices.foreach(println)
    //    println("************")
    //    positeGraph.edges.foreach(println)

    //    positeGraph.edges.foreach(println)

    val ppi = personalizedPositeItem(positeveEdgeRDD)
    ppi.cache()

    //    println("***********")
    //    ppi.foreach(println)

    //    println("---------------")

    //    val set = Set(1,4,5,6,8,10)
    //    set.subsets(2).map(_.toList).map(list=>(list(0),list(1))).foreach(println)
    //    println("------------")
    //var starTime = System.currentTimeMillis()

    val personalEdgesRDD = personlizedPreferenceItem(ppi.map(x=>x._2))

    personalEdgesRDD.cache()

    val tt = System.currentTimeMillis()
    val itemsRDD = (lines.map(i=>(i._2.toInt,-1)).join(cluster
    )).distinct().map(m=>(m._1.toLong,(m._2._2)))
    itemsRDD.cache()
    //val ttt = System.currentTimeMillis()
    //val tr = (ttt - tt)/1000.0
   // println(tr.toString())

    val itemGraph = Graph(itemsRDD,personalEdgesRDD)
    itemGraph.cache()
    //    itemGraph.vertices.foreach(println)
    //    itemGraph.edges.foreach(println)
    val resultfile = "./"+args(1)+args(0).toString+"recouserNum="+args(3)
      .toString+".txt"
    //val  file = new File(resultfile)
    val writer = new FileWriter(resultfile)
    writer.append("\n"+"-----------"+args(1)+"recouserNum:"+args(3)
      .toString+"iterator:"+args(5).toString()
      +"------------"+"\n")
    writer.append("\n"+"para cores= "+args(0).toString+"\n")
    //val tempTime = System.currentTimeMillis()
    //val t1 = (tempTime-startTime)/1000.0
    //println("begin  "+t1.toString())

    for (iterNum <- 1 to args(5).toInt) {
      writer.append("\n"+"iterator num====="+iterNum.toString+"\n")
      val recoUser = users.takeSample(true, args(3).toInt, 10).map(u => u._1)

      for (u <- recoUser) {

        val psg = PernonalizedSubGraph(u.toInt, ppi, itemGraph)

        recomendation(CacuDiversity(psg), args(4).toInt)

      }
      //val t2 = System.currentTimeMillis()
      //val t3 = (t2-t1)/1000.0
      //println("recom  "+t3.toString())

      val endTime = System.currentTimeMillis()
      val totalTime = (endTime - startTime) / 1000.0

      writer.write("\n"+"totaltime="+totalTime.toString()+"\n")
      writer.flush()

      startTime = System.currentTimeMillis()
    }
    writer.close()
    println("well done!")
    sc.stop()
  }
}
