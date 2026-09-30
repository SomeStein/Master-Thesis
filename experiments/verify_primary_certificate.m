function verify_primary_certificate(data_dir, toolbox_root)
% Check that the stored numerical certificate uses the tree drawn in the thesis.
    addpath(genpath(fullfile(toolbox_root,'ttoolboxes')),'-begin');
    loaded = load(fullfile(data_dir,'primary_certificate.mat'),'cert');
    cert = loaded.cert;
    assert(strcmp(cert.status,'certified') && cert.errorcode == ipa_errorcode.NOERROR);
    assert(isscalar(cert.JSR) && abs(cert.JSR-3) < 1e-9);
    assert(numel(cert.tree)==2);
    generators = {[2],[2;2]};
    expected = {{[-1;1]}, {[-2;1;1;1],[-2;1;1;2],[-2;1;2;1], ...
                 [-2;1;2;2],[-2;2;1]}};
    for t=1:2
        words = cert.tree{t}.leaf_words;
        assert(numel(words)==numel(expected{t}));
        for k=1:numel(expected{t})
            assert(any(cellfun(@(w) isequal(double(w(:)),expected{t}{k}),words)), ...
                'Stored leaf word differs from the tree in the thesis.');
        end
        assert(max(abs(eig(cert.tree{t}.generator_matrix))) <= 1+1e-9);
        expected_matrix=(cert.M{2}/3)^t;
        assert(norm(cert.tree{t}.generator_matrix-expected_matrix,'fro')<1e-9, ...
            'Stored generator matrix differs from the normalized matrix word.');
        assert(isequal(double(cert.tree{t}.leaf_words{1}(1)),-t));
        assert(all(cellfun(@(w) all(double(w(2:end))>=1 & ...
            double(w(2:end))<=2),words)));
        assert(isequal(double(generators{t}(:)),double(cert.hybrid_options{2}{t}(:))));
    end
    % Leaf trees are complete: every positive child of an internal positive
    % prefix is either another prefix, an uncovered leaf, or the covered
    % generator word. This includes the 2 and 22 branches, respectively.
    for t=1:2
        leaves = cellfun(@(w) double(w(2:end))',expected{t},'UniformOutput',false);
        prefixes = {[]};
        for k=1:numel(leaves)
            word=leaves{k};
            for j=1:numel(word)-1
                prefixes{end+1}=word(1:j); %#ok<AGROW>
            end
        end
        for k=1:numel(prefixes)
            prefix=prefixes{k};
            for letter=1:2
                child=[prefix letter];
                is_leaf=any(cellfun(@(w) isequal(w,child),leaves));
                is_prefix=any(cellfun(@(w) isequal(w,child),prefixes));
                is_covered=isequal(child,generators{t}(:)');
                assert(is_leaf || is_prefix || is_covered, ...
                    'The supplied word set leaves an uncovered branch.');
            end
        end
    end
    points=cell2mat(cert.norm);
    closed=sum(points(:)==ipa_constant.CLOSED_TREE_POINT);
    assert(closed==12, 'Unexpected number of closed test points.');
    vertices=cell2mat(cert.vertices);
    assert(rank(vertices)==size(cert.M{1},1),'Final vertices do not span R^d.');
    fprintf('Certificate checked: JSR %.16g; %d closed tree points; %d vertices; rank %d.\n', ...
        cert.JSR,closed,size(vertices,2),rank(vertices));
end
