sample_rate = 44100;
gain = 1581.1388300841895;
insig = randn(2, sample_rate);
insig = insig ./ max(abs(insig(:)));
outsig = MeddisHairCell(insig * gain,sample_rate,false);

save('../../../../data/test_data/meddishaircell.mat', 'insig', 'outsig');