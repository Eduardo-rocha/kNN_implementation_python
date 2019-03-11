%% plot data HW1 CS760

close all

data = [...
    1,0.9733096085409253;
    2,0.9635231316725978;
    3,0.9741992882562278;
    4,0.9706405693950177;
    5,0.9741992882562278;
    6,0.9733096085409253;
    7,0.9741992882562278;
    8,0.9750889679715302;
    9,0.9768683274021353;
    10,0.9724199288256228;
    11,0.9724199288256228;
    12,0.9733096085409253;
    13,0.9733096085409253;
    14,0.9715302491103203;
    15,0.9715302491103203;
    16,0.9715302491103203;
    17,0.9715302491103203;
    18,0.9697508896797153;
    19,0.9644128113879004;
    20,0.9661921708185054];

best = [9, 0.9777580071174378];

figure
plot(data(:,1), data(:,2), '*b')
title('Hyperparameter search')
grid on
xlabel('k')
% ylim([0.94,1])
ylabel('accuracy')
hold on
plot(best(1), best(2), '*r')
legend('validation set', 'test set')

%%
k10 = [
    337,0.8994661921708185;
    674,0.9368327402135231;
    1011,0.9537366548042705;
    1348,0.9590747330960854;
    1686,0.9661921708185054;
    2023,0.9697508896797153;
    2360,0.9715302491103203;
    2697,0.9733096085409253;
    3034,0.9724199288256228;
    3372,0.9733096085409253
    ];

k2 = [
    337,0.8879003558718861
    674,0.9323843416370107
    1011,0.9483985765124555
    1348,0.949288256227758
    1686,0.9572953736654805
    2023,0.9635231316725978
    2360,0.9644128113879004
    2697,0.9697508896797153
    3034,0.9715302491103203
    3372,0.9715302491103203
    ];

k4 = [
    337,0.9039145907473309;
    674,0.9555160142348754;
    1011,0.958185053380783;
    1348,0.9644128113879004;
    1686,0.9617437722419929;
    2023,0.9661921708185054;
    2360,0.9679715302491103;
    2697,0.9715302491103203;
    3034,0.9777580071174378;
    3372,0.9804270462633452
    ];

k6 = [
    337,0.9030249110320284;
    674,0.943950177935943;
    1011,0.9626334519572953;
    1348,0.9626334519572953;
    1686,0.9653024911032029;
    2023,0.9653024911032029;
    2360,0.9688612099644128;
    2697,0.9706405693950177;
    3034,0.9741992882562278;
    3372,0.9733096085409253
    ];

k8 = [
    337,0.8976868327402135;
    674,0.9395017793594306;
    1011,0.9537366548042705;
    1348,0.9626334519572953;
    1686,0.9653024911032029;
    2023,0.9715302491103203;
    2360,0.9715302491103203;
    2697,0.9741992882562278;
    3034,0.9777580071174378;
    3372,0.9768683274021353
    ];

k12 = [
    337,0.8959074733096085;
    674,0.9332740213523132;
    1011,0.9537366548042705;
    1348,0.9564056939501779;
    1686,0.9617437722419929;
    2023,0.9679715302491103;
    2360,0.9688612099644128;
    2697,0.9679715302491103;
    3034,0.9706405693950177;
    3372,0.9706405693950177
    ];

k14 = [
    337,0.8870106761565836;
    674,0.9279359430604982;
    1011,0.947508896797153;
    1348,0.9519572953736655;
    1686,0.9626334519572953;
    2023,0.9653024911032029;
    2360,0.9653024911032029;
    2697,0.9679715302491103;
    3034,0.9679715302491103;
    3372,0.9715302491103203
    ];




figure
hold on
plot(k2(:,1), k2(:,2))
plot(k6(:,1), k6(:,2))
plot(k10(:,1), k10(:,2))
plot(k14(:,1), k14(:,2))
title('Learning Curve')
grid on
xlabel('Training set size')
% ylim([0.94,1])
ylabel('Test accuracy')
legend('k = 2','k = 6','k = 10','k = 14')

%%
ROC20 = [
    0.0,0.7037037037037037;
    0.030303030303030304,0.7962962962962963;
    0.06060606060606061,0.9629629629629629;
    0.15151515151515152,0.9814814814814815;
    0.30303030303030304,1.0;
    1.0,1.0
    ];

ROC30 = [
    0.0,0.6481481481481481;
    0.030303030303030304,0.7962962962962963;
    0.06060606060606061,0.9444444444444444;
    0.09090909090909091,0.9629629629629629;
    0.15151515151515152,0.9814814814814815;
    0.42424242424242425,1.0
    1.0,1.0;
    ];

ROC25 = [
    0.0,0.6666666666666666;
    0.030303030303030304,0.7962962962962963;
    0.06060606060606061,0.9444444444444444;
    0.09090909090909091,0.9629629629629629;
    0.18181818181818182,0.9814814814814815;
    0.36363636363636365,1.0;
    1.0,1.0
    ];




figure
hold on
plot(ROC20(:,1), ROC20(:,2))
plot(ROC25(:,1), ROC25(:,2))
plot(ROC30(:,1), ROC30(:,2))
title('ROC Curve')
grid on
xlabel('FPR')
% ylim([0.94,1])
ylabel('TPR')
legend('k = 20','k = 25','k = 30')