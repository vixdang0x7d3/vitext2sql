CREATE TABLE `nhượng_quyền_thương_mại_của_các_đội` (
    `id_nhượng_quyền_thương_mại` TEXT,
    `tên_nhượng_quyền_thương_mại` TEXT,
    `hoạt_động` TEXT,
    `na_assoc` TEXT
);


CREATE TABLE `sân_vận_động` (
    `id_sân_vận_động` TEXT,
    `tên_sân_vận_động` TEXT,
    `tên_gọi_khác_của_sân_vận_động` TEXT,
    `thành_phố` TEXT,
    `tiểu_bang` TEXT,
    `quốc_gia` TEXT
);


CREATE TABLE `trường_đại_học` (
    `id_trường_đại_học` TEXT,
    `tên_đầy_đủ` TEXT,
    `thành_phố` TEXT,
    `tiểu_bang` TEXT,
    `quốc_gia` TEXT
);


CREATE TABLE `cầu_thủ` (
    `id_cầu_thủ` TEXT,
    `năm_sinh` INTEGER,
    `tháng_sinh` INTEGER,
    `ngày_sinh` INTEGER,
    `đất_nước_nơi_sinh` TEXT,
    `tiểu_bang_nơi_sinh` TEXT,
    `thành_phố_nơi_sinh` TEXT,
    `năm_mất` INTEGER,
    `tháng_mất` INTEGER,
    `ngày_mất` INTEGER,
    `đất_nước_nơi_mất` TEXT,
    `tiểu_bang_nơi_mất` TEXT,
    `thành_phố_nơi_mất` TEXT,
    `tên` TEXT,
    `họ` TEXT,
    `tên_đệm` TEXT,
    `cân_nặng` INTEGER,
    `chiều_cao` INTEGER,
    `tay_đánh_thuận` TEXT,
    `tay_ném_thuận` TEXT,
    `ngày_ra_mắt` TEXT,
    `ngày_chơi_trận_cuối` TEXT,
    `id_trong_retro` TEXT,
    `id_trong_brref` TEXT
);


CREATE TABLE `đội` (
    `năm` INTEGER,
    `id_giải_đấu` TEXT,
    `id_đội` TEXT,
    `id_nhượng_quyền_thương_mại` TEXT,
    `id_phân_hạng` TEXT,
    `thứ_hạng` INTEGER,
    `g` INTEGER,
    `số_lượng_trận_đấu_sân_nhà` INTEGER,
    `w` INTEGER,
    `l` INTEGER,
    `thắng_phân_hạng_hay_không` TEXT,
    `thắng_ở_giải_vô_địch_thế_giới_hay_không` TEXT,
    `thắng_ở_giải_bóng_chày_hay_không` TEXT,
    `thắng_ở_world_series_hay_không` TEXT,
    `r` INTEGER,
    `ab` INTEGER,
    `h` INTEGER,
    `loại_hai_người` INTEGER,
    `loại_ba_người` INTEGER,
    `hr` INTEGER,
    `bb` INTEGER,
    `so` INTEGER,
    `sb` INTEGER,
    `cs` INTEGER,
    `hbp` INTEGER,
    `sf` INTEGER,
    `ra` INTEGER,
    `er` INTEGER,
    `era` INTEGER,
    `cg` INTEGER,
    `sho` INTEGER,
    `sv` INTEGER,
    `ipouts` INTEGER,
    `ha` INTEGER,
    `hra` INTEGER,
    `bba` INTEGER,
    `soa` INTEGER,
    `e` INTEGER,
    `dp` INTEGER,
    `fp` INTEGER,
    `tên` TEXT,
    `sân_vận_động` TEXT,
    `số_lượng_dự_khán` INTEGER,
    `bpf` INTEGER,
    `ppf` INTEGER,
    `id_đội_trong_br` TEXT,
    `id_đội_trong_lahman45` TEXT,
    `id_đội_trong_retro` TEXT
);


CREATE TABLE `giải_đấu_sau_mùa_giải` (
    `năm` INTEGER,
    `vòng_đấu` TEXT,
    `id_đội_chiến_thắng` TEXT,
    `id_đội_chiến_thắng_tại_giải_đấu` TEXT,
    `id_đội_thua_cuộc` TEXT,
    `id_đội_thua_cuộc_tại_giải_đấu` TEXT,
    `thắng` INTEGER,
    `thua` INTEGER,
    `hoà` INTEGER
);


CREATE TABLE `bình_chọn_giải_thưởng_dành_cho_huấn_luận_viên` (
    `id_giải_thưởng` TEXT,
    `năm` INTEGER,
    `id_giải_đấu` TEXT,
    `id_cầu_thủ` TEXT,
    `số_điểm_chiến_thắng` INTEGER,
    `số_điểm_tối_đa` INTEGER,
    `lượt_bình_chọn_đầu_tiên` INTEGER
);


CREATE TABLE `bình_chọn_giải_thưởng_dành_cho_cầu_thủ` (
    `id_giải_thưởng` TEXT,
    `năm` INTEGER,
    `id_giải_đấu` TEXT,
    `id_cầu_thủ` TEXT,
    `số_điểm_chiến_thắng` INTEGER,
    `số_điểm_tối_đa` INTEGER,
    `lượt_bình_chọn_đầu_tiên` INTEGER,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);


CREATE TABLE `giải_thưởng_dành_cho_huấn_luyện_viên` (
    `id_cầu_thủ` TEXT,
    `id_giải_thưởng` TEXT,
    `năm` INTEGER,
    `id_giải_đấu` TEXT,
    `số_trận_hoà` TEXT,
    `ghi_chú` INTEGER,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);


CREATE TABLE `giải_thưởng_dành_cho_cầu_thủ` (
    `id_cầu_thủ` TEXT,
    `id_giải_thưởng` TEXT,
    `năm` INTEGER,
    `id_giải_đấu` TEXT,
    `số_trận_hoà` TEXT,
    `ghi_chú` TEXT,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);


CREATE TABLE `giải_đấu_của_tất_cả_các_ngôi_sao` (
    `id_cầu_thủ` TEXT,
    `năm` INTEGER,
    `số_lượng_trận_đấu` INTEGER,
    `id_trận_đấu` TEXT,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `gp` INTEGER,
    `vị_trí_bắt_đầu` INTEGER,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);


CREATE TABLE `cầu_thủ_của_trường_đại_học` (
    `id_cầu_thủ` TEXT,
    `id_trường_đại_học` TEXT,
    `năm` INTEGER,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`),
    FOREIGN KEY (`id_trường_đại_học`) REFERENCES `trường_đại_học`(`id_trường_đại_học`)
);


CREATE TABLE `thành_tích_đánh_bóng` (
    `id_cầu_thủ` TEXT,
    `năm` INTEGER,
    `thời_gian_chơi_bóng` INTEGER,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `g` INTEGER,
    `ab` INTEGER,
    `r` INTEGER,
    `h` INTEGER,
    `loại_hai_người` INTEGER,
    `loại_ba_người` INTEGER,
    `hr` INTEGER,
    `rbi` INTEGER,
    `sb` INTEGER,
    `cs` INTEGER,
    `bb` INTEGER,
    `so` INTEGER,
    `ibb` INTEGER,
    `hbp` INTEGER,
    `sh` INTEGER,
    `sf` INTEGER,
    `g_idp` INTEGER,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);


CREATE TABLE `thành_tích_ném_bóng` (
    `id_cầu_thủ` TEXT,
    `năm` INTEGER,
    `thời_gian_chơi_bóng` INTEGER,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `w` INTEGER,
    `l` INTEGER,
    `g` INTEGER,
    `gs` INTEGER,
    `cg` INTEGER,
    `sho` INTEGER,
    `sv` INTEGER,
    `ipouts` INTEGER,
    `h` INTEGER,
    `er` INTEGER,
    `hr` INTEGER,
    `bb` INTEGER,
    `so` INTEGER,
    `baopp` INTEGER,
    `era` INTEGER,
    `ibb` INTEGER,
    `wp` INTEGER,
    `hbp` INTEGER,
    `bk` INTEGER,
    `bfp` INTEGER,
    `gf` INTEGER,
    `r` INTEGER,
    `sh` INTEGER,
    `sf` INTEGER,
    `g_idp` INTEGER
);


CREATE TABLE `thành_tích_phòng_ngự` (
    `id_cầu_thủ` TEXT,
    `năm` INTEGER,
    `thời_gian_chơi_bóng` INTEGER,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `vị_trí` TEXT,
    `g` INTEGER,
    `gs` INTEGER,
    `số_lần_bị_loại_mỗi_lượt` INTEGER,
    `po` INTEGER,
    `a` INTEGER,
    `e` INTEGER,
    `dp` INTEGER,
    `pb` INTEGER,
    `wp` INTEGER,
    `sb` INTEGER,
    `cs` INTEGER,
    `zr` INTEGER,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);


CREATE TABLE `thành_tích_phòng_ngự_sân_ngoài` (
    `id_cầu_thủ` TEXT,
    `năm` INTEGER,
    `thời_gian_chơi_bóng` INTEGER,
    `glf` INTEGER,
    `gcf` INTEGER,
    `grf` INTEGER,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);


CREATE TABLE `thành_tích_huấn_luyện_viên` (
    `id_cầu_thủ` TEXT,
    `năm` INTEGER,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `ở_mùa_giải` INTEGER,
    `g` INTEGER,
    `w` INTEGER,
    `l` INTEGER,
    `thứ_hạng` INTEGER,
    `plyr_mgr` TEXT,
    FOREIGN KEY (`id_đội`) REFERENCES `đội`(`id_đội`)
);


CREATE TABLE `thành_tích_huấn_luyện_viên_theo_hiệp` (
    `id_cầu_thủ` TEXT,
    `năm` INTEGER,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `ở_mùa_giải` INTEGER,
    `hiệp_đấu` INTEGER,
    `g` INTEGER,
    `w` INTEGER,
    `l` INTEGER,
    `thứ_hạng` INTEGER,
    FOREIGN KEY (`id_đội`) REFERENCES `đội`(`id_đội`)
);


CREATE TABLE `lương` (
    `năm` INTEGER,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `id_cầu_thủ` TEXT,
    `lương` INTEGER
);


CREATE TABLE `đại_lộ_danh_vọng` (
    `id_cầu_thủ` TEXT,
    `id_năm` INTEGER,
    `được_bầu_chọn_bởi` TEXT,
    `số_lượng_bỏ_phiếu` INTEGER,
    `số_lượng_phiếu_cần` INTEGER,
    `số_lượng_phiếu_bầu` INTEGER,
    `được_lựa_chọn_hay_không` TEXT,
    `hạng_mục` TEXT,
    `lưu_ý_cần_thiết` TEXT,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);


CREATE TABLE `trận_đấu_sân_nhà` (
    `năm` INTEGER,
    `id_giải_đấu` TEXT,
    `id_đội` TEXT,
    `id_sân_vận_động` TEXT,
    `ngày_của_trận_đầu_tiên` TEXT,
    `ngày_của_trận_cuối_cùng` TEXT,
    `số_lượng_trận_đấu` INTEGER,
    `số_lần_mở_cửa` INTEGER,
    `số_lượng_dự_khán` INTEGER,
    FOREIGN KEY (`id_đội`) REFERENCES `đội`(`id_đội`),
    FOREIGN KEY (`id_sân_vận_động`) REFERENCES `sân_vận_động`(`id_sân_vận_động`)
);


CREATE TABLE `thành_tích_đội_theo_hiệp` (
    `năm` INTEGER,
    `id_giải_đấu` TEXT,
    `id_đội` TEXT,
    `hiệp_đấu` INTEGER,
    `id_phân_hạng` TEXT,
    `thắng_phân_hạng_hay_không` TEXT,
    `xếp_hạng` INTEGER,
    `g` INTEGER,
    `w` INTEGER,
    `l` INTEGER
);


CREATE TABLE `thành_tích_đánh_bóng_sau_mùa_giải` (
    `năm` INTEGER,
    `vòng_đấu` TEXT,
    `id_cầu_thủ` TEXT,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `g` INTEGER,
    `ab` INTEGER,
    `r` INTEGER,
    `h` INTEGER,
    `loại_hai_người` INTEGER,
    `loại_ba_người` INTEGER,
    `hr` INTEGER,
    `rbi` INTEGER,
    `sb` INTEGER,
    `cs` INTEGER,
    `bb` INTEGER,
    `so` INTEGER,
    `ibb` INTEGER,
    `hbp` INTEGER,
    `sh` INTEGER,
    `sf` INTEGER,
    `g_idp` INTEGER,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`),
    FOREIGN KEY (`id_đội`) REFERENCES `đội`(`id_đội`)
);


CREATE TABLE `thành_tích_ném_bóng_sau_mùa_giải` (
    `id_cầu_thủ` TEXT,
    `năm` INTEGER,
    `vòng_đấu` TEXT,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `w` INTEGER,
    `l` INTEGER,
    `g` INTEGER,
    `gs` INTEGER,
    `cg` INTEGER,
    `sho` INTEGER,
    `sv` INTEGER,
    `ipouts` INTEGER,
    `h` INTEGER,
    `er` INTEGER,
    `hr` INTEGER,
    `bb` INTEGER,
    `so` INTEGER,
    `baopp` TEXT,
    `era` INTEGER,
    `ibb` INTEGER,
    `wp` INTEGER,
    `hbp` INTEGER,
    `bk` INTEGER,
    `bfp` INTEGER,
    `gf` INTEGER,
    `r` INTEGER,
    `sh` INTEGER,
    `sf` INTEGER,
    `g_idp` INTEGER
);


CREATE TABLE `thành_tích_phòng_ngự_sau_mùa_giải` (
    `id_cầu_thủ` TEXT,
    `năm` INTEGER,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `vòng_đấu` TEXT,
    `vị_trí` TEXT,
    `g` INTEGER,
    `gs` INTEGER,
    `số_lần_bị_loại_mỗi_lượt` INTEGER,
    `po` INTEGER,
    `a` INTEGER,
    `e` INTEGER,
    `dp` INTEGER,
    `tp` INTEGER,
    `pb` INTEGER,
    `sb` INTEGER,
    `cs` INTEGER,
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);


CREATE TABLE `lần_xuất_hiện` (
    `năm` INTEGER,
    `id_đội` TEXT,
    `id_giải_đấu` TEXT,
    `id_cầu_thủ` TEXT,
    `g_toàn_bộ` INTEGER,
    `gs` INTEGER,
    `g_đánh_bóng` INTEGER,
    `g_phòng_ngự` INTEGER,
    `g_p` INTEGER,
    `g_c` INTEGER,
    `g_1b` INTEGER,
    `g_2b` INTEGER,
    `g_3b` INTEGER,
    `g_ss` INTEGER,
    `g_lf` INTEGER,
    `g_cf` INTEGER,
    `g_rf` INTEGER,
    `g_of` INTEGER,
    `g_dh` INTEGER,
    `g_ph` INTEGER,
    `g_pr` INTEGER,
    FOREIGN KEY (`id_đội`) REFERENCES `đội`(`id_đội`),
    FOREIGN KEY (`id_cầu_thủ`) REFERENCES `cầu_thủ`(`id_cầu_thủ`)
);
