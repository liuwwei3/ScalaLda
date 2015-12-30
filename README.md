#### HOW TO USE

1. 安装sbt, sbt官网：http://www.scala-sbt.org/

2. 安装完之后首先在次目录下运行，用来下载各种jar
```
sbt
```

3. 下载完之后编译lda
```
sbt package
```

4. 运行LDA：

```
spark-submit --class ScalaLda target/scala-2.10/scalalda_2.10-1.0.jar
```
