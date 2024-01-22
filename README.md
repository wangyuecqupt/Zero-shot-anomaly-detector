Zero-shot connector anomaly detector
===================================
  Pytorch implementation of 'A Zero-shot connector anomaly detection approach based on similarity-contrast learning'

  
Requirements
-----------------------------------
  The code has been tested with python 3.7, pytorch 1.7.1 and Cuda 10.1.
  
  	conda create -n zsad python=3.7.13
  	conda activate zsad
  	pip install -r requirements.txt
		
Required Data 
-----------------------------------
  To evaluate/train zsda, you will need to download the required datasets.
  
1.[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)<br />
2.[DeepPCB](https://github.com/Charmve/Surface-Defect-Detection/tree/master/DeepPCB)<br />
  
### Pretraining on VOC2012
    python ./train.py --bg_data path/to/VOC2012
        
### Evaluation on DeepPCB(This step only generates predicted bounding box results, and metric calculation needs to be done in [DeepPCB](https://github.com/tangsanli5201/DeepPCB)<br />.)
    python ./experiment_on_pcb.py --run_mode val --DeepPCB_path path/to/DeepPCB

### Finetune on DeepPCB(optional)  
    python ./experiment_on_pcb.py --run_mode train --DeepPCB_path path/to/DeepPCB

### Demo on image pair 
    python ./inference.py path/to/test_image path/to/template_image
    
### 比如我们可以在多行文本框里输入一段代码,来一个Java版本的HelloWorld吧
    public class HelloWorld {

      /**
      * @param args
	    */
	    public static void main(String[] args) {
		    System.out.println("HelloWorld!");

	    }

    }

### 链接
1.[点击这里你可以链接到www.google.com](http://www.google.com)<br />
2.[点击这里我你可以链接到我的博客](http://guoyunsky.iteye.com)<br />

###只是显示百度的图片
![baidu-images](http://www.baidu.com/img/bdlogo.png "baidu")  

###只是显示图片，这里用的是相对路径
![github-01.jpg](/images/01.jpg "github-01.jpg")

### 显示图片也可以用原生的html标签
<img src="http://su.bdimg.com/static/superplus/img/logo_white.png" />

###想点击某个图片进入一个网页,比如我想点击github的icorn然后再进入www.github.com
[![image]](http://www.github.com/)
[image]: /images/02.jpg "github-02.jpg"

### 文字被些字符包围
> 文字被些字符包围
>
> 只要再文字前面加上>空格即可
>
> 如果你要换行的话,新起一行,输入>空格即可,后面不接文字
> 但> 只能放在行首才有效

### 文字被些字符包围,多重包围
> 文字被些字符包围开始
>
> > 只要再文字前面加上>空格即可
>
>  > > 如果你要换行的话,新起一行,输入>空格即可,后面不接文字
>
> > > > 但> 只能放在行首才有效

### 部分文字的高亮
如果你想使一段话部分文字高亮显示，来起到突出强调的作用，那么可以把它用\`\`包围起来。
注意这不是单引号，而是Tab键和数字1键左边的按键（注意使用英文输入法）。<br />
	example：
		Thank`You`. Please `Call` Me `Coder`
### 代码片段高亮显示
GitHub的markdown语法还支持部分语言的代码片段高亮显示。只需要在代码的上一行和下一行用\`\`\`标记。
```Java
	public static void main(String[] args){} //Java
```
```c
	int main(int argc,char *argv[]) //C
```
```javascript
	document.getElementById("myH1").innerHTML="Welcome to my Homepage";//javascript
```
```cpp
	string &operator+(const string& A,const string& B) //cpp
```
	
### list列表条目使用
写文章时经常会用到list列表条目。GitHub的markdown语法里也支持使用圆点符。编辑的时候使用的是星号*。
* 国籍：中国
* 城市：北京
* 大学：清华大学 
<br/>注意：星号*后面要有一个空格。否则显示为普通星号。
GitHub还支持多级别的list列表条目：
* 编程语言
	* 脚本语言
		* Python

### 特殊字符处理
有一些特殊字符如<,#等,只要在特殊字符前面加上转义字符\即可<br />
你想换行的话其实可以直接用html标签\<br /\>
    

### 插入表格
在Markdown中插入表格比较麻烦，需要Markdown的扩展语法，但是插入HTML就没有那么麻烦了，因此我们可以通过曲线救国的方式来插入表格。       
在Markdown中，`&`符号和`<`会自动转换成HTML。

	<div>
	    <table border="0">
		  <tr>
		    <th>one</th>
		    <th>two</th>
		  </tr>
		  <tr>
		    <td>Hello</td>
		    <td>你好</td>
		  </tr>
	    </table>
	</div>
	
<div>
        <table border="0">
	  <tr>
	    <th>one</th>
	    <th>two</th>
	  </tr>
	  <tr>
	    <td>Hello</td>
	    <td>你好</td>
	  </tr>
	</table>
</div>
