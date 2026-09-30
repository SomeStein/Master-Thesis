function run_ipa_comparison(toolbox_root, output_dir, repetitions, case_filter, repeat_filter, mode_filter)
%RUN_IPA_COMPARISON Reproduce the off/strict IPA comparison in the thesis.
% Toolbox: ttoolboxes commit c67fe5e, with toolbox-c67fe5e-leaf-words.patch.
% Usage: run_ipa_comparison(toolbox_root, output_dir, 3)

    if nargin < 3, repetitions = 3; end
    if nargin < 4, case_filter = ''; end
    if nargin < 5, repeat_filter = 0; end
    if nargin < 6, mode_filter = ''; end
    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile(fileparts(mfilename('fullpath')), 'data');
    end
    if nargin < 1 || isempty(toolbox_root)
        toolbox_root = getenv('IPA_TOOLBOX_ROOT');
    end
    assert(~isempty(toolbox_root), 'Set IPA_TOOLBOX_ROOT or pass the toolbox checkout path.');
    assert(repetitions >= 1 && repetitions == floor(repetitions));
    addpath(genpath(fullfile(toolbox_root, 'ttoolboxes')), '-begin');
    assert(startsWith(which('ipa'), toolbox_root), 'The requested toolbox is not first on the MATLAB path.');
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end

    cases = make_cases();
    write_manifest(output_dir, toolbox_root, repetitions, cases);
    filename = fullfile(output_dir, 'ipa_runs.csv');
    is_new = ~isfile(filename);
    fid = fopen(filename, 'a');
    assert(fid > 0, 'Could not open result CSV.');
    cleaner = onCleanup(@() fclose(fid));
    if is_new
        fprintf(fid, ['case_id,category,dimension,num_matrices,repeat,mode,elapsed_s,' ...
            'jsr_lower,jsr_upper,errorcode,status,iterations,vertices,tree_closed_points,error_identifier\n']);
    end

    common = {'plot','none','maxiter',80,'maxtime',120,'numcore',0, ...
        'basictest',0,'solver','gurobi','v',-1};
    for i = 1:numel(cases)
        ex = cases(i);
        if ~isempty(case_filter) && ~strcmp(ex.id,case_filter), continue; end
        for rep = 1:repetitions
            if repeat_filter ~= 0 && rep ~= repeat_filter, continue; end
            if mod(rep,2) == 1, modes = {'off','strict'};
            else, modes = {'strict','off'}; end
            for j = 1:2
                mode = modes{j};
                if ~isempty(mode_filter) && ~strcmp(mode,mode_filter), continue; end
                args = [common ex.options];
                t = tic;
                jsr = [nan nan]; nfo = struct(); error_id = '';
                try
                    if strcmp(mode, 'strict')
                        [jsr,nfo] = ipa(ex.M,args{:},'hybrid','strict',ex.hybrid_options{:});
                    else
                        [jsr,nfo] = ipa(ex.M,args{:},'hybrid','off');
                    end
                catch me
                    error_id = me.identifier;
                    fprintf(2,'%s repeat %d %s: %s: %s\n',ex.id,rep,mode,me.identifier,me.message);
                end
                elapsed = toc(t);
                errorcode = get_log(nfo,'errorcode',nan);
                iterations = get_log(nfo,'iteration',nan);
                [vertices,closed] = vertex_counts(nfo);
                lower = min(real(jsr(:))); upper = max(real(jsr(:)));
                status = classify(errorcode,jsr,error_id);
                fprintf(fid,'%s,%s,%d,%d,%d,%s,%.17g,%.17g,%.17g,%.17g,%s,%.17g,%.17g,%.17g,%s\n', ...
                    ex.id,ex.category,size(ex.M{1},1),numel(ex.M),rep,mode,elapsed, ...
                    lower,upper,errorcode,status,iterations,vertices,closed,error_id);
                fprintf('%s %d %s: %s, JSR [%g,%g], %g s\n', ...
                    ex.id,rep,mode,status,lower,upper,elapsed);
                if strcmp(ex.id,'mejstrikpumm257') && rep == 1 && strcmp(mode,'strict')
                    save_primary_certificate(output_dir,ex,jsr,nfo,args,errorcode,status);
                end
            end
        end
    end
    clear cleaner
end

function cases = make_cases()
    cases = struct('id',{},'category',{},'M',{},'options',{},'hybrid_options',{});
    for k = 251:257
        id = sprintf('mejstrikpumm%d',k);
        ex = struct('id',id,'category','gallery','M',{gallery_matrixset(id)}, ...
            'options',{{}},'hybrid_options',{{}});
        if k == 257
            ex.hybrid_options = {'generator',{[2],[2;2]}, ...
                'leafs',{'minimal',{[1 1 1],[1 1 2],[1 2 1],[1 2 2],[2 1]}}};
        end
        cases(end+1) = ex; %#ok<AGROW>
    end
    ex = struct('id','mejstrik_reif_2x2','category','published', ...
        'M',{{diag([1,-1]),[0 1; 1/2 0]}},'options',{{}},'hybrid_options',{{}});
    cases(end+1) = ex;
    settings = [2 3 2 .35 .45 .25; 2001 7 4 .24 .42 .04; 2033 7 4 .42 .42 .04];
    for k = 1:size(settings,1)
        s = settings(k,:);
        [M,Q] = controlled_matrices(s(1),s(2),s(3),s(4),s(5),s(6));
        id = sprintf('controlled_d%d_n%d_seed%d',s(2),s(3),s(1));
        ex = struct('id',id,'category','controlled','M',{M}, ...
            'options',{{}},'hybrid_options',{{}});
        % The initial extra vertices are standard basis columns 2 through d.
        extra = eye(s(2)); extra = extra(:,2:end);
        ex.options = {'smp',{int32(1)},'v0',{Q(:,1)},'lambda',1, ...
            'extravertex',extra,'autoextravertex',0,'balancing',0, ...
            'invariantsubspace','none','normalization','off','algorithm','r'};
        cases(end+1) = ex; %#ok<AGROW>
    end
end

function [M,Q] = controlled_matrices(seed,dim,num_matrix,alpha,scale,coupling)
% Exact construction from ipa_hybrid_discover_examples.m at commit c67fe5e.
    rng(seed,'twister');
    [Q,R] = qr(randn(dim));
    Q = Q * diag(sign(diag(R)) + (diag(R) == 0));
    stable = alpha * (0.75 .^ (0:dim-2));
    D = diag([1 stable]);
    G = real(Q * D / Q);
    M = cell(1,num_matrix); M{1} = G;
    for j = 2:num_matrix
        B = zeros(dim);
        B(1,1) = 0.35 + 0.05*j;
        for r = 2:dim
            B(r,r) = scale * stable(r-1) / max(alpha,eps);
            B(1,r) = coupling * (0.7 + 0.1*j) / r;
            B(r,1) = coupling * (-1)^r / (r+j);
        end
        B = scale * B;
        M{j} = real(Q * B / Q);
    end
end

function value = get_log(nfo,name,default)
    value = default;
    if isfield(nfo,'log') && isfield(nfo.log,name), value = double(nfo.log.(name)); end
end

function [vertices,closed] = vertex_counts(nfo)
    vertices = nan; closed = nan;
    if ~isfield(nfo,'blockvar') || isempty(nfo.blockvar), return; end
    ct = nfo.blockvar{1}.cyclictree;
    if isfield(ct,'VV'), vertices = size([ct.VV{:}],2);
    elseif isfield(ct,'V'), vertices = size([ct.V{:}],2); end
    if isfield(ct,'norm')
        vals = [ct.norm{:}];
        closed = sum(vals(:) == ipa_constant.CLOSED_TREE_POINT);
    end
end

function status = classify(code,jsr,error_id)
    if ~isempty(error_id), status = 'exception'; return; end
    if ismember(code,[-10,-20,-40,-4,-3]) && isscalar(jsr) && isfinite(jsr)
        status = 'certified';
    elseif code == 120, status = 'iteration_limit';
    elseif ismember(code,[70,75]), status = 'time_limit';
    elseif ismember(code,[90,95,100]), status = 'resource_limit';
    else, status = 'stopped'; end
end

function save_primary_certificate(output_dir,ex,jsr,nfo,args,errorcode,status)
    cert = struct('M',{ex.M},'JSR',jsr,'options',{args}, ...
        'hybrid_options',{ex.hybrid_options},'errorcode',errorcode,'status',status);
    if isfield(nfo,'blockvar') && ~isempty(nfo.blockvar)
        ct = nfo.blockvar{1}.cyclictree;
        if isfield(ct,'hybrid'), cert.tree = ct.hybrid.tree; end
        if isfield(ct,'norm'), cert.norm = ct.norm; end
        if isfield(ct,'VV'), cert.vertices = ct.VV; end
    end
    save(fullfile(output_dir,'primary_certificate.mat'),'cert','-v7');
end

function write_manifest(output_dir,toolbox_root,repetitions,cases)
    filename = fullfile(output_dir,'run_manifest.txt');
    fid = fopen(filename,'w'); assert(fid > 0); cleaner = onCleanup(@() fclose(fid));
    [~,commit] = system(sprintf('git -C "%s" rev-parse HEAD',toolbox_root));
    fprintf(fid,'toolbox_commit=%s',commit);
    fprintf(fid,'toolbox_patch=toolbox-c67fe5e-leaf-words.patch\n');
    fprintf(fid,'matlab=%s\n',version);
    fprintf(fid,'architecture=%s\n',computer);
    fprintf(fid,'solver=gurobi\n');
    fprintf(fid,'ipa_path=%s\n',which('ipa'));
    fprintf(fid,'gurobi_path=%s\n',which('gurobi'));
    fprintf(fid,'repetitions=%d\n',repetitions);
    fprintf(fid,'common_options=plot none; maxiter 80; maxtime 120; numcore 0; basictest 0; solver gurobi; verbose -1\n');
    fprintf(fid,'strict_options_for_mejstrikpumm257=generator {[2],[2;2]}; leafs {minimal,{[1 1 1],[1 1 2],[1 2 1],[1 2 2],[2 1]}}\n');
    for k = 1:numel(cases), fprintf(fid,'case=%s\n',cases(k).id); end
    clear cleaner
end
