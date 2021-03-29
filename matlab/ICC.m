% revision
clc; clear all; close all;
addpath('Y:\Research\Pancancer\Nature Machine Intelligence Revision\Revise\Violinplot-Matlab-master');

load('ICC.mat');
boxplot(ICC);
ylabel('ICC');
ylim([0, 1]);
