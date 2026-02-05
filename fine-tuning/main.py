import os
import torch
from torch.utils.data import DataLoader, random_split
from data import NetworkFlowDataset, load_flow_data,TransformerPacketDataset,get_balanced_masked_bert_dataset
from data import balance_and_augment_data,augment_and_balance,CombinedFlowDataset
from models import TextFeatureExtractor,FusionModel,UnifiedFlowModel
from trainer import FlowTrainer
from transformers import BertTokenizer
import numpy as np
import swanlab
from transformers import get_scheduler
import torch.nn as nn
from sklearn.model_selection import train_test_split
import argparse
import torch_optimizer as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# 配置
config = {
    "bert_path": "/root/netgpt/model",
    "data_path": "/root/autodl-tmp/data/USTC-TFC2016-master/Malware/",
    "tokenizer_path":"/root/netgpt/vocab/bert_tokenizer",
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 32,
    "text_lr": 3e-5,
    "trans_lr": 8e-5,
    "fusion_lr":5e-6,
    "epochs": 30,
    "feature_num":28,
    "target_count":500,#每个类别设定多少数量
    "trans_max_epoch":30,
    "text_max_epoch" :10,
    "fusion_max_epoch":20,
    "trans_patience":10,
    "trans_hidden_size":256,
    "fusion_hidden_size":256,
    "bert_use_multiclassifier":False,
    "use_parallel":False,
    "start_freeze_layer":8,
    "use_unfreeze":False,
    "continue_train":False
}

def save_model(model, optimizer,scheduler, epoch, model_name="model.pth",use_parallel = False):
    if use_parallel:
        """ 保存模型和优化器的状态 """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.module.state_dict(),
            'scheduler_state_dict': scheduler.module.state_dict(),  # ✅ 保存 scheduler
        }
    else:
        """ 保存模型和优化器的状态 """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # ✅ 保存 scheduler
        }
    torch.save(checkpoint, model_name)
    # print(f"Model saved to {model_name}")


def load_model(model, optimizer,scheduler,device, model_name="model.pth",use_parallel = False):
    """ 加载模型和优化器的状态 """
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
     #将优化器状态迁移到cuda
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    start_epoch = checkpoint['epoch']
    print(f"Model loaded from {model_name}")
    return start_epoch,model,optimizer,scheduler

def main():
    start_epoch = 0
    device = config["device"]

    # 截取这些样本
    data = np.load(config["data_path"]+"/splitcap/all_feature.npz",allow_pickle=True)
    packet_sequences = data["sequences"]
    flow_labels = data["labels"]
    flow_features = data["stat_features"]
    # # 加载 flow-level 统计特征数据
    # flow_data = np.load(config["data_path"]+"splitcap/all_featueres.npz")
    # flow_features = flow_data["features"]
    # flow_labels = flow_data["labels"]
    
    # # 一致性校验（可选）
    # print("Label 分布：", np.unique(packet_labels, return_counts=True))
    # print("Label 分布：", np.unique(flow_labels, return_counts=True))
    # assert (packet_labels == flow_labels).all(), "两个模态的标签不一致"
    
    # transformer数据集
    X1_bal,  y_bal = augment_and_balance(flow_features,flow_labels, target_count=config["target_count"], noise_level=0.01)
    X2_bal,  y_bal = balance_and_augment_data(packet_sequences, flow_labels, config["target_count"], noise_level=0.05, mask_prob=0.1, shuffle_prob=0.1)
    # 划分训练验证集
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1_bal, X2_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=3407
    )
    trans_train_set = CombinedFlowDataset(X1_train, X2_train, y_train)
    trans_val_set = CombinedFlowDataset(X1_val, X2_val, y_val)
    
    trans_train_loader = DataLoader(trans_train_set,batch_size=config["batch_size"], shuffle=False,drop_last = True)
    trans_val_loader = DataLoader(trans_val_set, batch_size=config["batch_size"],shuffle=False,drop_last = True)


    samples, labels, label_dict = load_flow_data(config["data_path"])
    bert_y = [y for x,y in samples]
    print("Label 分布：", np.unique(flow_labels, return_counts=True))
    print("Label 分布：", np.unique(bert_y, return_counts=True))
    assert (flow_labels == bert_y).all(), "两个模态的标签不一致"

    # bert数据集
    num_classes = len(labels)
    if num_classes > 20:#如果类别数大于20，使用自定义分类器
        config["bert_use_multiclassifier"] = True
    print(f"Total num classe is {num_classes}")
    

    tokenizer = BertTokenizer.from_pretrained(config["tokenizer_path"])
    new_samples = get_balanced_masked_bert_dataset(
        samples, tokenizer=tokenizer,
        target_count=config["target_count"], mask_prob=0.2
    )
    bert_train, bert_val = train_test_split(
        new_samples ,test_size=0.2, stratify=y_bal, random_state=3407
    )
    train_set = NetworkFlowDataset(bert_train, tokenizer) #"input_ids""attention_mask" "label"
    val_set = NetworkFlowDataset(bert_val, tokenizer)
    
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False,drop_last = True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,drop_last = True)

    # 模型初始化
    #加载bertmodel,transformer model,和fusion model
    text_model  = TextFeatureExtractor(config["bert_path"], num_classes,config["bert_use_multiclassifier"],dropout_prob = 0.5,start_freeze_layer = config["start_freeze_layer"])

    trans_model = UnifiedFlowModel(stat_feat_dim = 42, seq_feat_dim=28, seq_len=100, hidden_dim=config["trans_hidden_size"], num_classes=num_classes)
    
    fusion_model = FusionModel(hidden_size=config["fusion_hidden_size"], trans_hidden_size = config["trans_hidden_size"],num_classes=num_classes,dropout = 0.3)

    #加载optimizer
    # 确保优化器只更新可训练参数
    optimizer_grouped_parameters = [
    {"params": text_model.bert.bert.encoder.layer[:8].parameters(), "lr": config["text_lr"]/5},
    {"params": text_model.bert.bert.encoder.layer[8:].parameters(), "lr": config["text_lr"]},
    {"params": text_model.bert.classifier.parameters(), "lr": config["text_lr"]*2},
]
    text_optim = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    trans_optim = torch.optim.AdamW(
        trans_model.parameters(),
        weight_decay=0.01,
        lr=config["trans_lr"],
    )

    fusion_optim = torch.optim.AdamW(
                fusion_model.parameters(),
                lr=config["fusion_lr"]
            )
    #学习策略
    num_training_steps = config["epochs"] * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    text_scheduler = get_scheduler(
        name="cosine",
        optimizer=text_optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    trans_scheduler = get_scheduler(
        name="cosine",
        optimizer=trans_optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    fusion_scheduler = get_scheduler(
        name="cosine",
        optimizer=fusion_optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    #如果从上一轮开始接着训练
    if config["continue_train"]:
        start_epoch,text_model,text_optim,text_scheduler = load_model(text_model, text_optim,text_scheduler, config["device"],model_name = config["data_path"]+"/splitcap/text_model.pth",use_parallel = config["use_parallel"])
        start_epoch,trans_model,trans_optim,trans_scheduler = load_model(trans_model, trans_optim,trans_scheduler, config["device"],model_name = config["data_path"]+"/splitcap/trans_model.pth",use_parallel = config["use_parallel"])
        start_epoch,fusion_model,fusion_optim,fusion_scheduler = load_model(fusion_model, fusion_optim, fusion_scheduler,config["device"],model_name = config["data_path"]+"/splitcap/fusion_model.pth",use_parallel = config["use_parallel"])
    #如果要使用多GPU训练
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个 GPU")
        config["use_parallel"] = True
        text_model  = nn.DataParallel(text_model)
        trans_model = nn.DataParallel(trans_model)
        fusion_model = nn.DataParallel(fusion_model)
        
    text_model = text_model.to(device)
    trans_model = trans_model.to(device)
    fusion_model = fusion_model.to(device)


    # 初始化SwanLab实验
    swanlab.init(
        project = "bert-triformer",
        experiment_name=config["data_path"].split("/")[-3] +"-" +config["data_path"].split("/")[-2],
        config={
            "model_type": "fusion_model",
            "device": str(device)
        }
    )
    #如果之前有激活好的bert直接使用
    if os.path.exists(config["data_path"]+"/splitcap/fintuned_bert.pth"):
        text_model.load_state_dict(torch.load(config["data_path"]+"/splitcap/fintuned_bert.pth"))
    trainer = FlowTrainer(num_classes,text_model, trans_model,fusion_model,text_optim,trans_optim,fusion_optim,text_scheduler,trans_scheduler,fusion_scheduler, device,config["use_parallel"],config["start_freeze_layer"],swanlab)
    #如果接着之前的训练
    if config["continue_train"]:
        text_model.enable_fintuning(11)
        trainer = FlowTrainer(num_classes,text_model, trans_model,fusion_model,text_optim,trans_optim,fusion_optim,text_scheduler,trans_scheduler,fusion_scheduler, device,config["use_parallel"],config["start_freeze_layer"],swanlab)
        for epoch in range(start_epoch,start_epoch+config["epochs"]):
            train_metrics, _, _,text_model,trans_model,fusion_model = trainer.train_epoch(
                train_loader,
                trans_train_loader
            )
    
            val_metrics, preds, labels = trainer.evaluate(val_loader,trans_val_loader)
    
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            save_model(text_model, text_optim,text_scheduler, epoch, model_name=config["data_path"]+"/splitcap/text_model.pth",use_parallel = config["use_parallel"])
            if trans_model:
                save_model(trans_model, trans_optim, trans_scheduler,epoch, model_name=config["data_path"]+"/splitcap/trans_model.pth",use_parallel = config["use_parallel"])
            if fusion_model:
                save_model(fusion_model, fusion_optim,fusion_scheduler, epoch, model_name=config["data_path"]+"/splitcap/fusion_model.pth",use_parallel = config["use_parallel"])
            # if (epoch)%5==0 or epoch == config["epochs"]:
            #     save_model(text_model, text_optim,text_scheduler, epoch, model_name=config["data_path"]+"/splitcap/text_model.pth",use_parallel = config["use_parallel"])
            #     if trans_model:
            #         save_model(trans_model, trans_optim, trans_scheduler,epoch, model_name=config["data_path"]+"/splitcap/trans_model.pth",use_parallel = config["use_parallel"])
            #     if fusion_model:
            #         save_model(fusion_model, fusion_optim,fusion_scheduler, epoch, model_name=config["data_path"]+"/splitcap/fusion_model.pth",use_parallel = config["use_parallel"])
    #如果从头训练
    else:
        if os.path.exists(config["data_path"]+"/splitcap/fintuned_bert.pth"):
            text_model.load_state_dict(torch.load(config["data_path"]+"/splitcap/fintuned_bert.pth"))
        trainer = FlowTrainer(num_classes,text_model, trans_model,fusion_model,text_optim,trans_optim,fusion_optim,text_scheduler,trans_scheduler,fusion_scheduler, device,config["use_parallel"],config["start_freeze_layer"],swanlab)
        trainer.warmup_trans_model(trans_train_loader,max_epochs=config["trans_max_epoch"],val_loader=trans_val_loader, patience=config["trans_patience"])
        # if not os.path.exists(config["data_path"]+"/splitcap/fintuned_bert.pth"):
        #     trainer.warmup_text_model(train_loader, max_epochs=config["text_max_epoch"],val_loader=val_loader, patience=config["trans_patience"])
        #     if config["use_parallel"]:
        #         torch.save(text_model.module.state_dict(), config["data_path"]+"/splitcap/fintuned_bert.pth")
        #     else:
        #         torch.save(text_model.state_dict(), config["data_path"]+"/splitcap/fintuned_bert.pth")
        trainer.warmup_fusion_model(train_loader, trans_train_loader, max_epochs=config["fusion_max_epoch"])

        # 创建epoch级别的进度条
        epoch_pbar = tqdm(total=config["epochs"], desc="Training Progress", initial=start_epoch)
        
        for epoch in range(start_epoch, start_epoch + config["epochs"]):
            train_metrics, _, _, text_model, trans_model, fusion_model = trainer.train_epoch(
                train_loader,
                trans_train_loader
            )
        
            val_metrics, preds, labels = trainer.evaluate(val_loader, trans_val_loader)
            
            # 更新进度条信息
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'Epoch': f'{epoch + 1}/{config["epochs"]}',
                'Train Loss': f'{train_metrics["loss"]:.4f}',
                'Val Loss': f'{val_metrics["loss"]:.4f}',
                'Val Acc': f'{val_metrics["accuracy"]:.4f}',
                'Val F1': f'{val_metrics["f1"]:.4f}'
            })
            
            # 每5个epoch或最后一个epoch输出详细指标
            if (epoch + 1) % 5 == 0 or epoch == start_epoch + config["epochs"] - 1:
                tqdm.write(f"\n=== Epoch {epoch + 1}/{config['epochs']} Detailed Metrics ===")
                tqdm.write(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
                tqdm.write(f"Val Accuracy: {val_metrics['accuracy']:.4f} | Precision: {val_metrics['precision']:.4f}")
                tqdm.write(f"Val Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
                
            
                # 保存模型
                save_model(text_model, text_optim, text_scheduler, epoch, 
                           model_name=config["data_path"]+"/splitcap/text_model.pth", 
                           use_parallel=config["use_parallel"])
                if trans_model:
                    save_model(trans_model, trans_optim, trans_scheduler, epoch, 
                               model_name=config["data_path"]+"/splitcap/trans_model.pth", 
                               use_parallel=config["use_parallel"])
                if fusion_model:
                    save_model(fusion_model, fusion_optim, fusion_scheduler, epoch, 
                               model_name=config["data_path"]+"/splitcap/fusion_model.pth", 
                               use_parallel=config["use_parallel"])
        
        epoch_pbar.close()
        # 绘制混淆矩阵
        cm = confusion_matrix(labels, preds)
        print(cm)
        print("Training completed!")

        # for epoch in range(start_epoch,start_epoch+config["epochs"]):
        #     train_metrics, _, _,text_model,trans_model,fusion_model = trainer.train_epoch(
        #         train_loader,
        #         trans_train_loader
        #     )
    
        #     val_metrics, preds, labels = trainer.evaluate(val_loader,trans_val_loader)
    
        #     print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        #     print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        #     print(f"Val Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
        #     save_model(text_model, text_optim,text_scheduler, epoch, model_name=config["data_path"]+"/splitcap/text_model.pth",use_parallel = config["use_parallel"])
        #     if trans_model:
        #         save_model(trans_model, trans_optim, trans_scheduler,epoch, model_name=config["data_path"]+"/splitcap/trans_model.pth",use_parallel = config["use_parallel"])
        #     if fusion_model:
        #         save_model(fusion_model, fusion_optim,fusion_scheduler, epoch, model_name=config["data_path"]+"/splitcap/fusion_model.pth",use_parallel = config["use_parallel"])

    swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ServiceVPN", action="store_true", help="Use Service-VPN Datasets")    
    parser.add_argument("--ServiceNonVPN", action="store_true", help="Use Service-NonVPN Datasets")    
    parser.add_argument("--AppVPN", action="store_true", help="Use App-VPN Datasets")
    parser.add_argument("--AppNonVPN", action="store_true", help="Use App-NonVPN Datasets")
    parser.add_argument("--Tor", action="store_true", help="Use Tor Datasets")
    parser.add_argument("--NonTor", action="store_true", help="Use NonTor Datasets")
    parser.add_argument("--Benign", action="store_true", help="Use Benign Datasets")
    parser.add_argument("--Malware", action="store_true", help="Use Malware Datasets")
    parser.add_argument("--Flood", action="store_true", help="Use Flood Datasets")
    parser.add_argument("--RTSPBruteForce", action="store_true", help="Use RTSPBruteForce Datasets")
    parser.add_argument("--datacon2020", action="store_true", help="Use datacon2020 Datasets")
    parser.add_argument("--datacon2021part1", action="store_true", help="Use datacon2021part1 Datasets")
    parser.add_argument("--datacon2021part2", action="store_true", help="Use datacon2021part2 Datasets")
    parser.add_argument("--CrossPlatformandroid", action="store_true", help="Use CrossPlatformandroid Datasets")
    parser.add_argument("--CrossPlatformios", action="store_true", help="Use CrossPlatformios Datasets")
    parser.add_argument("--NUDT", action="store_true", help="Use NUDT Datasets")
    
    args = parser.parse_args()
    if args.ServiceVPN:
        config["data_path"] = "/root/autodl-tmp/berttrans/ISCX-VPN-Service/VPN/"
    
    if args.ServiceNonVPN:
        config["data_path"] = "/root/autodl-tmp/berttrans/ISCX-VPN-Service/NonVPN/"
    
    if args.AppVPN:
        config["data_path"] = "/root/autodl-tmp/berttrans/ISCX-VPN-App/VPN/"

    if args.AppNonVPN:
        config["data_path"] = "/root/autodl-tmp/berttrans/ISCX-VPN-App/NonVPN/"
    
    if args.Tor:
        config["data_path"] = "/root/autodl-tmp/berttrans/ISCX-Tor/Tor/"

    if args.NonTor:
        config["data_path"] = "/root/autodl-tmp/berttrans/ISCX-Tor/NonTor/"

    if args.Benign:
        config["data_path"] = "/root/autodl-tmp/berttrans/USTC-TFC2016/Benign/"

    if args.Malware:
        config["data_path"] = "/root/autodl-tmp/berttrans/USTC-TFC2016/Malware/"

    if args.Flood:
        config["data_path"] = "/root/autodl-tmp/data/CIC_IOT_Dataset2022_Attacks/Flood/"

    if args.RTSPBruteForce:
        config["data_path"] = "/root/autodl-tmp/data/CIC_IOT_Dataset2022_Attacks/RTSP-Brute-Force/"

    if args.datacon2020:
        config["data_path"] = "/root/autodl-tmp/data/datacon2020/train/"

    if args.datacon2021part1:
        config["data_path"] = "/root/autodl-tmp/data/datacon2021/part1/test/"

    if args.datacon2021part2:
        config["data_path"] = "/root/autodl-tmp/data/datacon2021/part2/train/"

    if args.CrossPlatformandroid:
        config["data_path"] = "/root/autodl-tmp/data/CrossPlatform/android/"

    if args.CrossPlatformios:
        config["data_path"] = "/root/autodl-tmp/data/CrossPlatform/ios/"
    
    if args.NUDT:
        config["data_path"] = "/root/autodl-tmp/data/NUDT/merge/"
    main()

    
