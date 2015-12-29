import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.util.Random
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import java.io._

object ScalaLda{
    

    //configs
    val MAX_ITR = 30
    val E_CON_THRES = 1e-6
    val K = 2
    val Alpha = 3.0 / K
    val EM_ITR = 40
    val DATA = "/app/st/wise-tc/liuweiwei02/data1"
    //val DATA = "/app/st/wise-tc/liuweiwei02/LDA_training_doc3/*"
    val PARTITION = 300
    //var beta_global = Array(Array(1.0f,2.0f),Array(1.0f,2.0f))
    
    
    def digamma(_x: Double): Double = {
        val x = _x + 6.0
        var p = 1.0 / ( x * x )
        p=(((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
        p=p+math.log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0)
        return p
    }
    
    def lgamma(_x: Double): Double = {
        var z = 1.0 / ( _x * _x )
        val x= _x + 6.0
        z=(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/x
        z=(x-0.5)*math.log(x)-x+0.918938533204673+z-math.log(x-1)-math.log(x-2)-math.log(x-3)-math.log(x-4)-math.log(x-5)-math.log(x-6)
        return z
    }
    
    def compute_likelihood(word_ids: Array[Int], word_counts: Array[Double], beta: Map[Int, Array[Float]], phi: Array[Array[Double]], var_gamma: Array[Double], di_gamma : Array[Double]): Double = {
        val gamma_sum = var_gamma.sum
        val digamma_sum = digamma(gamma_sum)
        var res = lgamma( Alpha * K ) - K * lgamma(Alpha) - lgamma(gamma_sum)
        for (k <- 0 until K){
            res = res + ( Alpha - var_gamma(k) ) * (di_gamma(k) - digamma_sum) + lgamma(var_gamma(k))
            for (n <- 0 until word_ids.length){
                res = res + word_counts(n) * (phi(n)(k) * ( 
                di_gamma(k) - digamma_sum - math.log(phi(n)(k)) + math.log(beta(word_ids(n))(k))  ))
            }
        }
        return res
    }

    def normal_line(line :(Int, Array[Float])) : (Int, Array[Float]) = {
        val res = new Array[Float]( K )
        var sum = 0.0f
        for (i <- 0 until K){
            sum = sum + line._2(i)
        }
        for (i <- 0 until K){
            res(i) = line._2(i) / sum
        }
        return (line._1, res)
    }

    def normalize(line: (Int, Array[Float]), line_sum: Array[Float]): (Int, Array[Float]) = {
        val res = new Array[Float](line_sum.length)
        for (i <- 0 until line_sum.length){
            res(i) = line._2(i) / line_sum(i)
        }
        return (line._1, res)
    }

    def arr_add(a: Array[Float], b: Array[Float]): Array[Float] = {
        val res = for(i<- 0 until a.length) yield {a(i) + b(i)}
        return res.toArray
    }
    
    def print_beta(beta: Array[Array[Float]]) {
        for (i<- 0 until beta.length){
            for (j <- 0 until beta(i).length){
                print(beta(i)(j))
                print(" ")
            }
            print("\n")
        }
    }

    def print_beta2(beta: Map[Int, Array[Float]]) {
        for (i<- beta.keys){
            for (j <- 0 until beta(i).length){
                print(beta(i)(j))
                print(" ")
            }
            print("\n")
        }
    }
    
    def update_beta(input: Array[(Int, Array[Float])]) : Array[Array[Float]] = {
        val ids = for (i <- 0 until input.length) yield { input(i)._1 }
        var res = new Array[Array[Float]](ids.max + 1)
        for (i <- 0 until input.length){
            res(input(i)._1) = input(i)._2
        }
        return res
    }
    
    def init_beta(args: Array[String]): (Int, Array[Array[Float]]) = {
        if (args.length != 3){
            return (0, Array( Array(1.0f,2.0f), Array(1.0f,2.0f) ) )
        }
        val start = args(0).toInt
        var res = new Array[Array[Float]](args(2).toInt + 1)
        for (line <- Source.fromFile(args(1)).getLines ){
            val temp = line.trim().split(" ")
            if (temp.length == K+1){
                val temp2 = for(k <- 0 until K) yield {temp(k+1).toFloat}
                res(temp(0).toInt) = temp2.toArray
            }
        }
        return (start, res)
    }

    def save_model(i: Int, beta_arr: Array[Array[Float]]){
        val out = new DataOutputStream(  new BufferedOutputStream(new FileOutputStream("beta."+i.toString)))
        out.writeInt(K)
        for (line <- beta_arr){
            if (line!=null){
                for (i <- 0 until line.length){
                    if (i >= 0){
                        out.writeFloat(line(i))
                    }
                }
            }
        }
        out.close()
    }
    
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("SparkLda")
        val sc = new SparkContext(conf)
        val text = sc.textFile(DATA).repartition(PARTITION).cache()
        var likelihood, likelihood_old = 0.0f
        //var beta_arr = Array( Array(1.0f,2.0f), Array(1.0f,2.0f) )
        var (start, beta_arr )= init_beta(args)
        for (i<- start until EM_ITR){
            var beta_global = sc.broadcast(beta_arr)
            println("broadcast success, id:", beta_global.id)
            def Expectation(line: String, itr_num: Int) : Array[(Int, Array[Float])] = {
                val terms = line.trim.split(' ') 
                val real_terms = ArrayBuffer[String]()
                for(i <- 0 until terms.length){
                    if (!terms(i).trim.isEmpty()){
                        real_terms += terms(i).trim
                    }
                }
                val N = real_terms.length
                if (N <= 0){
                    return new Array[(Int, Array[Float])](0)
                }
                val word_ids = new Array[Int](N)
                val word_counts = new Array[Double](N)
                for(i <- 0 until N){
                    val ele = real_terms(i).split(':')
                        word_ids(i) = ele(0).toInt
                        word_counts(i) = ele(1).toDouble * K // multipy K to bigger gamma for accuracy
                }
                val total_words = word_counts.sum
                val var_gamma = new Array[Double](K)
                val di_gamma = new Array[Double](K)
                for (i <- 0 until K){
                    var_gamma(i) = Alpha + total_words / K
                    di_gamma(i) = digamma(var_gamma(i))
                }
                var beta : Map[Int, Array[Float]] = Map()
                var ITR_NUM = MAX_ITR
                if (itr_num > 0){
                    for (n<- 0 until N){
                        beta += (word_ids(n) -> beta_global.value( word_ids(n) ) )
                        //beta += (word_ids(n) -> beta_global( word_ids(n) ) )
                    }
                 }
                 else{
                     ITR_NUM = 1
                     var r = new Random()
                         for (n<- 0 until N){
                             var random_list = for (i <- 0 until K) yield {r.nextFloat()+0.1f}
                             beta += (word_ids(n) -> random_list.toArray )
                         }
                 }
                 val phi = Array.ofDim[Double](N, K)
                 var converged = 1.0
                 var likelihood_old, likelihood = 0.0
                 var i = 0
                 while (i < ITR_NUM && converged > E_CON_THRES){
                     for (n <- 0 until N){
                         for (k <- 0 until K){
                             phi(n)(k) = math.max( 1e-45 , math.exp(di_gamma(k))) * math.max( 1e-45, beta(word_ids(n))(k) )
                         }
                         val phi_sum = phi(n).sum
                         for (k <- 0 until K){
                             phi(n)(k) = phi(n)(k) / phi_sum
                         }
                     }
                     for (k <- 0 until K){
                         var temp = 0.0
                         for (n <- 0 until N){
                             temp = temp + word_counts(n) * phi(n)(k)
                         }
                         var_gamma(k) = Alpha + temp
                         di_gamma(k) = digamma(var_gamma(k))
                     }
                     likelihood = compute_likelihood(word_ids, word_counts, beta, phi, var_gamma, di_gamma)
                     converged = math.abs((likelihood_old - likelihood) / (likelihood_old + 1e-10))
                     likelihood_old = likelihood
                     i = i+1
                 }

                 val res = new Array[(Int, Array[Float])](N)
                 for (n<- 0 until N){
                     val temp = for(k<- 0 until  K) yield { math.max(1e-45f, phi(n)(k).toFloat) } 
                     res(n) = (word_ids(n), temp.toArray)
                 }
                 return res
            }
            var beta = text.flatMap(line => Expectation(line, i) ).reduceByKey(arr_add).map(line => normal_line(line) )
            beta.cache()
            val line_sum = beta.flatMap(line => for(i<- 0 until line._2.length) yield {(i, line._2(i))} ).reduceByKey((a,b) => a+b).collect()
            var line_sum2 = for (ele <- line_sum.sortWith(_._1<_._1)) yield {ele._2}
            val new_beta = beta.map(line => normalize(line, line_sum2))
            val beta_result =  new_beta.collect() 
            println("Iteration finished, ", i,"beta result length ..." ,beta_result.length)
            beta_arr = update_beta( beta_result )
            println("Generate new beta finished, ", i,"beta length ..." ,beta_arr.length)
            try{beta.unpersist(false)}catch{ case e : Exception =>{println("Exception while unpersisting beta"+i.toString)}}
            try{
                beta_global.destroy()
            }catch{
                case e : Exception => {
                    println("Exception while destroying beta_global, i = " + i.toString)
                }
            }

            save_model(i, beta_arr)
        }
    }

}
