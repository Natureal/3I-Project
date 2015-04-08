#include "highgui.h"
#include "cv.h"
#include<math.h>
#include<iostream>
using namespace std;
using namespace cv;
const double PI = acos(-1.0);
void cvThresholdOtsu(IplImage* src, IplImage* dst);
void cvSkinOtsu(IplImage* src, IplImage* dst);//yCbCr
void cvFind(IplImage *img,IplImage *dst1,IplImage *dst2);
void hand_and_face_detect(IplImage *temp,IplImage *src_image,CvMemStorage *storage,CvSeq *contours,CvPoint &pt1,CvPoint &pt2);
int max(int i,int j,int k,int l);
int min(int i,int j,int k,int l);
void Skin_HSV_new(IplImage *img,IplImage *dst);
bool choose(IplImage *img);

//==================================================================================
struct Gesture_judge{
    inline void Draw_line(IplImage *graph,int row){
        printf("drawline : %d\n",row);
        int W = graph->width;
        for(int j = 0; j < W; ++j){
            cvSet2D(graph,row,j,Scalar(255,0,0));
        }
        row = row - 1;
        if(row < 0) return;
        for(int j = 0; j < W; ++j){
            cvSet2D(graph,row,j,Scalar(255,0,0));
        }
    }

    inline void Draw_point(IplImage *graph,int row,int col){
        int W = graph->width;
        int H = graph->height;
        for(int i = -2; i <= 2; ++i)
        for(int j= -2; j <= 2; ++j){
            if(row + i < 0 || row + i >= H || col + j < 0 || col + j >= W) continue;
            cvSet2D(graph,row + i,col + j,Scalar(0,0,255));
        }
    }

    inline void Draw_center(IplImage *graph,int &row,int &col,int val){
        CvPoint cen;
        cen.x = row,cen.y = col;
        CvScalar color;
        color.val[0] = val;
        cvCircle(graph,cen,10,color,2,8,0);
    }

    inline void Find_mid(IplImage *graph,int row,int &mid,int &left,int &right){
        int W = graph->width;
        const int cnt_part = 30;
        int w_part = W / cnt_part;
        int white_part[cnt_part + 5] = {0};
        for(int i = 1; i <= cnt_part; ++i){
            int t_right = i * w_part,cnt_white = 0;
            for(int j = (i - 1) * w_part; j < t_right; ++j){
                CvScalar tmp = cvGet2D(graph,row,j);
                if(tmp.val[0] == 255) cnt_white++;
            }
            if(cnt_white > 0.9 * w_part) white_part[i] = 1;
            else white_part[i] = 0;
        }

        int cur = 0,suc_max = 0,max_pos = -1;
        for(int i = 1; i <= cnt_part; ++i){
            if(white_part[i]) cur++;
            else cur = 0;
            if(cur > suc_max){
                suc_max = cur;
                max_pos = i;
            }
        }

        left = w_part * (max_pos - suc_max);
        right = w_part * (max_pos);
        mid = left + (right - left) / 2;
    }

    void Find_line(IplImage *graph,int &top_x,int &max_x,int &bot_x,int &up_x){
        top_x = max_x = bot_x = up_x = -1;
        int white[10000] = {0},H = graph->height,W = graph->width;
        int white_max = -1;
        for(int i = 0; i < H; i+=3){
            for(int j = 0; j < W; ++j){
                CvScalar tmp = cvGet2D(graph,i,j);
                if(tmp.val[0] == 255) white[i]++;
            }
            /*if(i < H - 6){
                if(white[i] > (int)(W * 0.35)){
                    bot_x = bot_x == -1 ? i : bot_x;
                }
            }*/
            if(up_x == -1 && white[i] > W / 20) up_x = i;
        }
        for(int i = 0; i < H; i+=3){
            if(white[i] > white_max){
                white_max = white[i];
                max_x = i;
            }
        }
        for(int i = max_x; i < H; i+=3){
            if(white[i] > (int)(white_max * 0.6)){
                bot_x = i;
            }
        }
        for(int i = max_x; i > 0; i-=3){
            if(white[i] > (int)(white_max * 0.8)){
                top_x = i;
            }
        }
        //确定top线的x值
        //top_x = max_x + 0.8 * (max_x - bot_x);
    }

    int Find_inter(IplImage *graph,IplImage *bgroud,CvPoint cen,int r){
        int W = graph->width,H = graph->height;
        const int cnt_part = 50;
        int ang_part = 360 / cnt_part;
        int white[cnt_part + 5] ={0};
        for(int i = 1; i <= cnt_part; ++i){
            int top = i * ang_part,cnt_white = 0;
            for(int b = (i - 1) * ang_part; b < top; ++b){
                int tx = abs(cen.x + r * cos(b * PI / 180));
                int ty = abs(cen.y + r * sin(b * PI / 180));
                if(tx < 0 || tx >= H || ty < 0 || ty >= W) continue;
                if(b == (i - 1) * ang_part)
                    cvLine(bgroud,cen,cvPoint(tx,ty),cvScalar(255,0,0));
                CvScalar tmp = cvGet2D(graph,tx,ty);
                if(tmp.val[0] == 255){
                    cnt_white++;
                }
            }
            printf("cnt_white of %d : %d\n",i,cnt_white);
            if(cnt_white >= 2) white[i] = 1;
        }
        for(int i = 1; i <= cnt_part; ++i){
            cout << white[i] << " , ";
        }
        puts("");
        return 0;
    }

    void Solve(IplImage *graph,IplImage *bgroud){
        int top_x,top_y,top_left,top_right; //AB线
        int max_x,max_y,max_left,max_right; //CD线
        int bot_x,bot_y,bot_left,bot_right; //EF线
        int up_x; //手掌顶部的y轴坐标
        Find_line(graph,top_x,max_x,bot_x,up_x);
        Find_mid(graph,top_x,top_y,top_left,top_right);
        Find_mid(graph,max_x,max_y,max_left,max_right);
        Find_mid(graph,bot_x,bot_y,bot_left,bot_right);

        int cir_x,cir_y,cir_r = 0,tr,p1,p2,p3,p4,difx_1,dify_1,difx_2,dify_2;
        difx_1 = top_x - max_x;
        dify_1 = top_y - max_y;
        difx_2 = max_x - bot_x;
        dify_2 = max_y - bot_y;
        for(int i = 1; i <= 1; ++i){
            int tx = max_x + difx_1 / i;
            int ty = max_y + dify_1 / i;
            //Find_radius(graph,tx,ty,tr);
            p1 = abs(up_x - tx); //center -> up
            p2 = abs(tx - bot_x); //center -> bot
            p3 = 1e9;//abs(tx - left_x); //center -> left
            p4 = 1e9;//abs(right_x - tx); //center -> right
            tr = min(p1,min(p2,min(p3,p4)));
            if(tr > cir_r){
                cir_r = tr;
                cir_x = tx;
                cir_y = ty;
            }
        }
        for(int i = 1; i <= 1; ++i){
            int tx = bot_x + difx_2 / i;
            int ty = bot_y + dify_2 / i;
            //Find_radius(graph,tx,ty,tr);
            p1 = abs(up_x - tx); //center -> up
            p2 = abs(tx - bot_x); //center -> bot
            p3 = 1e9;//abs(tx - left_x); //center -> left
            p4 = 1e9;//abs(right_x - tx); //center -> right
            tr = min(p1,min(p2,min(p3,p4)));
            if(tr > cir_r){
                cir_r = tr;
                cir_x = tx;
                cir_y = ty;
            }
        }

        cout << "cir_r : " << cir_x << " , " << cir_y << endl;

        CvPoint cen;
        //cen.x = cir_x,cen.y = cir_y;
        cen.x = (max_left + max_right) / 2;
        cen.y = (bot_x + (up_x - bot_x) / 3);
        Draw_center(graph,cen.x,cen.y,0);
        CvScalar color;
        color.val[0] = 0;
        int cnt_inter = 0;
        for(int i = 0; i <= 0; ++i){
            cvCircle(graph,cen,cir_r * (1 + 0.1 * i),color,2,8,0);
            cnt_inter = max(cnt_inter,Find_inter(graph,bgroud,cen,cir_r));
        }

        //draw angle line
        CvPoint tar;
        for(int i = 1; i < 12; ++i){
            tar.x = cen.x - 2 * cir_r * cos(i * PI/12);
            tar.y = cen.y - 2 * cir_r * sin(i * PI/12);
            //cvLine(graph,cen,tar,cvScalar(0));
        }

        //debug draw line / mid point

        Draw_line(bgroud,up_x);
        Draw_line(bgroud,top_x);
        Draw_line(bgroud,max_x);
        Draw_line(bgroud,bot_x);

        cout << "top_pos : " << top_x << " ; top_mid : " << top_y << endl;
        cout << "max_pos : " << max_x << " ; max_mid : " << max_y << endl;
        cout << "bot_pos : " << bot_x << " ; bot_mid : " << bot_y << endl;

        Draw_point(bgroud,top_x,top_y);
        Draw_point(bgroud,top_x,top_left);
        Draw_point(bgroud,top_x,top_right);

        Draw_point(bgroud,max_x,max_y);
        Draw_point(bgroud,max_x,max_left);
        Draw_point(bgroud,max_x,max_right);

        Draw_point(bgroud,bot_x,bot_y);
        Draw_point(bgroud,bot_x,bot_left);
        Draw_point(bgroud,bot_x,bot_right);

        //debug
    }

}gesture_judge;
//====================================================================================

void cvThresholdOtsu(IplImage* src, IplImage* dst);
void cvSkinOtsu(IplImage* src, IplImage* dst);//yCbCr
void cvFind(IplImage *img,IplImage *bkg);
void hand_and_face_detect(IplImage *temp,IplImage *src_image,CvMemStorage *storage,CvSeq *contours,CvPoint &pt1,CvPoint &pt2);
int max(int i,int j,int k,int l);
int min(int i,int j,int k,int l);
void Skin_HSV_new(IplImage *img,IplImage *dst);
bool choose(IplImage *img);
bool hand_detect(IplImage *cmp,CvPoint &pt1,CvPoint &pt2);
void Update_Bkg(IplImage *curr_Img,IplImage *bg_Img,IplImage *cmp);
//void reset
int main()
{
    //const char* filename = "E:\\_实验室\\肤色识别代码\\一般\\facedetect\\图片素材\\11.jpg";
    IplImage* image = cvLoadImage( "a5.JPG", 1 );
    IplImage *bkg=cvLoadImage("a6.JPG",1);
    if( image )
    {
        //IplImage *dst1=cvCreateImage(cvGetSize(image),8,1);
        //IplImage *dst2=cvCreateImage(cvGetSize(image),8,1);

        //cvWaitKey(0);

        cout << "graph 's width : " << image->width << endl;
        cout << "graph 's height : " << image->height << endl;

        cvFind(image,bkg);
        cvShowImage("result",image);
        //cvShowImage("HSV",HSV);
        cvWaitKey(0);
        cvReleaseImage( &image );
        return 0;

    }
    else
    {
        cout<<"cannot loard img!";
        return -1;
    }
}
void cvFind(IplImage *img,IplImage *bkg)
{
    int sum;
    int i=0;
    IplImage *cmp=cvCreateImage(cvGetSize(img),8,1);
    //cvShowImage("cmp",cmp);
    Update_Bkg(img,bkg,cmp);
    //cvShowImage("cmp",cmp);
    IplImage *dst=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
     IplImage *dst1=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
    CvMemStorage *storage = cvCreateMemStorage(0);
    CvSeq *contours =0;//存储提取的轮廓图像
    CvPoint pt1,pt2;
    cvSkinOtsu(img,dst);
    cvCopy(dst,dst1,NULL);
    cvFindContours(dst,storage,&contours,sizeof(CvContour),CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);//找所有连通域的轮廓
    for(;contours!=0;contours=contours->h_next)     //当有轮廓的时候，一直调用函数，找手和脸
    {
        hand_and_face_detect(dst,img,storage,contours,pt1,pt2);
            /*cout<<pt1.x<<"  "<<pt1.y;
            cout<<pt2.x<<"  "<<pt2.y;*/
            /*cout<<"\n";*/
        if(pt1.x==0&&pt2.y==0)
        {
            continue;
        }
        if(!hand_detect(cmp,pt1,pt2))
            continue;
        cvSetImageROI(img, cvRect(pt1.x+5,pt1.y,abs(pt2.x-pt1.x-15),abs(pt2.y-pt1.y)));
        //感兴趣区域大小不一样
        IplImage *img2 = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
        //拷贝
        cvCopy(img, img2, NULL);

        //重置
        cvResetImageROI(img);

        IplImage *dst = cvCreateImage(cvGetSize(img2),8,1);
        Skin_HSV_new(img2,dst);
        gesture_judge.Solve(dst,img2);
        cvShowImage("img2",img2);
        cvShowImage("dst",dst);

        cvWaitKey(0);
        //cvShowImage("dst",dst1);
    }

}
void hand_and_face_detect(IplImage *temp,IplImage*src_image,CvMemStorage *storage,CvSeq *contours,CvPoint &pt1,CvPoint &pt2)
{
        CvBox2D rect=cvMinAreaRect2(contours,storage);//求连通轮廓的最小外包矩形

        CvPoint2D32f rect_pts0[4];
        cvBoxPoints(rect, rect_pts0);

        //因为cvPolyLine要求点集的输入类型是CvPoint**
        //所以要把 CvPoint2D32f 型的 rect_pts0 转换为 CvPoint 型的 rect_pts
        //并赋予一个对应的指针 *pt
        int npts = 4,k=0;
        CvPoint rect_pts[4], *pt = rect_pts;
        int area=0;

        for (int i=0; i<4; i++)     //转换坐标形式
        {
            rect_pts[i]= cvPointFrom32f(rect_pts0[i]);
        }

        pt1.x=min(rect_pts[0].x,rect_pts[1].x,rect_pts[2].x,rect_pts[3].x);//算一个斜的矩形的最小的正的外接矩形，左下角坐标
        pt1.y=min(rect_pts[0].y,rect_pts[1].y,rect_pts[2].y,rect_pts[3].y);

        pt2.x=max(rect_pts[0].x,rect_pts[1].x,rect_pts[2].x,rect_pts[3].x);//右上角坐标
        pt2.y=max(rect_pts[0].y,rect_pts[1].y,rect_pts[2].y,rect_pts[3].y);
        if(pt2.x>=temp->width)
            pt2.x=temp->width;
        if(pt2.y>=temp->height)
            pt2.y=temp->height;
        area=abs(rect_pts[0].x-rect_pts[2].x)*(rect_pts[0].y-rect_pts[2].y);
        if(area<0.025*(src_image->width*src_image->height))   //判断最小包围盒的面积，如果过小，忽略
        {
            pt1.x=0;
            pt2.y=0;
            //cout<<area;
            return;
        }

        //画出Box
        else
        {
            //cvRectangle(src_image,pt1,pt2,CV_RGB(255,0,0));//画矩形
            if(area>0.2*(src_image->width*src_image->height))
            {
                pt1.x=pt1.x+0.52*abs(pt1.x-pt2.x);
            }
            cvRectangle(src_image,pt1,pt2,CV_RGB(255,0,0));//画矩形
            //cout<<area;
        }
}

void cvThresholdOtsu(IplImage* src, IplImage* dst)
{
    int height=src->height;
    int width=src->width;

    //histogram
    float histogram[256]= {0};
    for(int i=0; i<height; i++)
    {
        unsigned char* p=(unsigned char*)src->imageData+src->widthStep*i;
        for(int j=0; j<width; j++)
        {
            histogram[*p++]++;
        }
    }
    //normalize histogram
    int size=height*width;
    for(int i=0; i<256; i++)
    {
        histogram[i]=histogram[i]/size;
    }

    //average pixel value
    float avgValue=0;
    for(int i=0; i<256; i++)
    {
        avgValue+=i*histogram[i];
    }

    int threshold;
    float maxVariance=0;
    float w=0,u=0;
    for(int i=0; i<256; i++)
    {
        w+=histogram[i];
        u+=i*histogram[i];

        float t=avgValue*w-u;
        float variance=t*t/(w*(1-w));
        if(variance>maxVariance)
        {
            maxVariance=variance;
            threshold=i;
        }
    }

    cvThreshold(src,dst,threshold,255,CV_THRESH_BINARY);
}
void cvSkinOtsu(IplImage* src, IplImage* dst)//yCbCr
{
    assert(dst->nChannels==1&& src->nChannels==3);

    IplImage* ycrcb=cvCreateImage(cvGetSize(src),8,3);
    IplImage* cr=cvCreateImage(cvGetSize(src),8,1);
    cvCvtColor(src,ycrcb,CV_BGR2YCrCb);
    cvSplit(ycrcb,0,cr,0,0);

    cvThresholdOtsu(cr,cr);
    cvCopyImage(cr,dst);
    cvReleaseImage(&cr);
    cvReleaseImage(&ycrcb);
    cvSmooth(dst,dst,CV_MEDIAN);
    if(choose(dst))
        return;
    else
        Skin_HSV_new(src,dst);
}
void Skin_HSV_new(IplImage *img,IplImage *dst)
{
    //cvEqualizeHist(img,img); //直方图均衡
    IplImage *img_t =cvCreateImage(cvGetSize(img),8,3);
    cvCvtColor(img,img_t, CV_BGR2HSV);
    for(int i=0;i<img_t->width;i++)
    {
        for(int j=0;j<img_t->height;j++)
        {
            CvScalar temp=cvGet2D(img_t,j,i);
            int value = (((temp.val[1]+temp.val[2])*1.0)/temp.val[0]);
            if(value<9)
            {
                *(dst->imageData+j*dst->widthStep+i)=0;
            }
            else
            {
                *(dst->imageData+j*dst->widthStep+i)=255;
            }
            //cout<<value<<'\n';
        }
    }

    for(int i = 1; i>=0;i--)
    {
        cvDilate(dst,dst);
        //cvErode(dst,dst);
    }
    for(int i = 1; i>=0;i--)
    {
        cvErode(dst,dst);
    }
    //cvShowImage("www",dst);
    //cvSmooth(dst,dst);
    cvReleaseImage(&img_t);
}

bool choose(IplImage *img)
{
    int sum=0;
    for(int i =0;i<img->width;i++)
    {
        for(int j=0;j<img->height;j++)
        {
            CvScalar temp=cvGet2D(img,j,i);
            int value=temp.val[0];
            if(value == 255)
                sum++;
        }
    }
    if (sum<0.15*(img->width*img->height)||sum>0.85*(img->width*img->height))
    {
        //cout<<sum;
        return false;
    }
    else
        return true;
}

int max(int i,int j,int k,int l)//找四个数字中的最大值
{
    int max;
    if(i>=j)
        max=i;
    else
        max=j;
    if(max<=k)
        max=k;
    if(max<=l)
        max=l;
    return max;
}
int min(int i,int j,int k,int l)//找四个数字中的最小值
{
    int min;
    if(i<=j)
        min=i;
    else
        min=j;
    if(min>=k)
        min=k;
    if(min>=l)
        min=l;
    return min;
}





void Update_Bkg(IplImage *curr_Img,IplImage *bg_Img,IplImage *cmp)
{
    IplImage *curr_Img_t = cvCreateImage(cvGetSize(curr_Img),8,1);
    IplImage *temp = cvCreateImage(cvGetSize(curr_Img),8,1);
    //cvShowImage("temp",temp);
    IplImage *bg_Img_t = cvCreateImage(cvGetSize(curr_Img),8,1);
    cvCvtColor(curr_Img,curr_Img_t,CV_BGR2GRAY);
    cvCvtColor(bg_Img,bg_Img_t,CV_BGR2GRAY);
    CvScalar Avg,Sdv;
    //double avg,sdv;
    cvAvgSdv(curr_Img_t,&Avg,&Sdv,0);
    int var = Sdv.val[0]*Sdv.val[0];
    int iCount = 0;

    for(int i=0;i<curr_Img->width;i++)
    {
        for(int j=0;j<curr_Img->height;j++)
        {
            int s=cvGet2D(curr_Img_t,j,i).val[0];
            int v=cvGet2D(bg_Img_t,j,i).val[0];
            int difference=(int)abs(s-v);
            if(difference>=0.48*Sdv.val[0]&&difference>=0.02*var)
            {
                iCount++;
                *(temp->imageData+j*temp->widthStep+i) = 0;
            }
            else
                *(temp->imageData+j*temp->widthStep+i) = 255;
            if(iCount>curr_Img_t->width*curr_Img_t->height*0.2)
            {
                cout<<"BG Change!";
                return;
            }
        }
    }
    cvDilate(temp,temp);
    cvDilate(temp,temp);
    cvShowImage("window",temp);
    cvCopyImage(temp,cmp);
    //cvWaitKey(0);
    cvReleaseImage(&temp);
    cvReleaseImage(&curr_Img_t);
    cvReleaseImage(&bg_Img_t);
}

bool hand_detect(IplImage *cmp,CvPoint &pt1,CvPoint &pt2)
{
    int m = abs(pt1.x-pt2.x);
    int n =abs( pt1.y-pt2.y);
    /*cout<<pt2.y;
    cout<<cmp->width;
    cout<<cmp->height;*/
    //cvWaitKey(0);
    int sum = 0;
    for(int i=pt1.x;i<pt2.x-2;i++)
    {
        for(int j=pt1.y;j<pt2.y-2;j++)
        {
            int value;
            value=cvGet2D(cmp,j,i).val[0];
            //cout<<value;
            //cout<<"hh";
            if(value==0)
                sum++;
        }
    }
    //cout<<m*n<<'\t';
    //cout<<'\n'<<sum<<'\t';
    if(sum>=0.02*m*n)
    {
        //cout<<"kk";
        return true;
    }
    else
        return false;
}

